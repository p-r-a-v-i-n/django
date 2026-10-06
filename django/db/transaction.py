import warnings
from asyncio import get_running_loop
from contextlib import ContextDecorator, contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import iscoroutinefunction

from django.db import (
    DEFAULT_DB_ALIAS,
    DatabaseError,
    Error,
    NotSupportedError,
    ProgrammingError,
    connections,
)
from django.utils.deprecation import RemovedInDjango2028Warning
from django.utils.warnings import django_file_prefixes

_async_atomic_blocks = ContextVar("django_async_atomic_blocks", default={})


def _uses_async_transaction(connection):
    if _async_atomic_blocks.get().get(connection.alias):
        return True
    if not connection.features.supports_async or connection.in_atomic_block:
        return False
    try:
        get_running_loop()
    except RuntimeError:
        return False
    return True


class TransactionManagementError(ProgrammingError):
    """Transaction management is used improperly."""

    pass


def get_connection(using=None):
    """
    Get a database connection by name, or the default database connection
    if no name is provided. This is a private API.
    """
    if using is None:
        using = DEFAULT_DB_ALIAS
    return connections[using]


def get_autocommit(using=None):
    """Get the autocommit status of the connection."""
    return get_connection(using).get_autocommit()


def set_autocommit(autocommit, using=None):
    """Set the autocommit status of the connection."""
    return get_connection(using).set_autocommit(autocommit)


def commit(using=None):
    """Commit a transaction."""
    get_connection(using).commit()


def rollback(using=None):
    """Roll back a transaction."""
    get_connection(using).rollback()


def savepoint(using=None):
    warnings.warn(
        "savepoint() is deprecated. Use savepoint_create() instead.",
        category=RemovedInDjango2028Warning,
        skip_file_prefixes=django_file_prefixes(),
    )
    return savepoint_create(using=using)


def savepoint_create(using=None):
    """
    Create a savepoint (if supported and required by the backend) inside the
    current transaction. Return an identifier for the savepoint that will be
    used for the subsequent rollback or commit.
    """
    return get_connection(using).savepoint()


def savepoint_rollback(sid, using=None):
    """
    Roll back the most recent savepoint (if one exists). Do nothing if
    savepoints are not supported.
    """
    get_connection(using).savepoint_rollback(sid)


def savepoint_commit(sid, using=None):
    """
    Commit the most recent savepoint (if one exists). Do nothing if
    savepoints are not supported.
    """
    get_connection(using).savepoint_commit(sid)


def clean_savepoints(using=None):
    """
    Reset the counter used to generate unique savepoint ids in this thread.
    """
    get_connection(using).clean_savepoints()


def get_rollback(using=None):
    """Get the "needs rollback" flag -- for *advanced use* only."""
    connection = get_connection(using)
    if _uses_async_transaction(connection):
        return connection.get_async_rollback()
    return connection.get_rollback()


def set_rollback(rollback, using=None):
    """
    Set or unset the "needs rollback" flag -- for *advanced use* only.

    When `rollback` is `True`, trigger a rollback when exiting the innermost
    enclosing atomic block that has `savepoint=True` (that's the default). Use
    this to force a rollback without raising an exception.

    When `rollback` is `False`, prevent such a rollback. Use this only after
    rolling back to a known-good state! Otherwise, you break the atomic block
    and data corruption may occur.
    """
    connection = get_connection(using)
    if _uses_async_transaction(connection):
        return connection.set_async_rollback(rollback)
    return connection.set_rollback(rollback)


@contextmanager
def mark_for_rollback_on_error(using=None):
    """
    Internal low-level utility to mark a transaction as "needs rollback" when
    an exception is raised while not enforcing the enclosed block to be in a
    transaction. This is needed by Model.save() and friends to avoid starting a
    transaction when in autocommit mode and a single query is executed.

    It's equivalent to:

        connection = get_connection(using)
        if connection.get_autocommit():
            yield
        else:
            with transaction.atomic(using=using, savepoint=False):
                yield

    but it uses low-level utilities to avoid performance overhead.
    """
    try:
        yield
    except Exception as exc:
        connection = get_connection(using)
        if connection.in_atomic_block:
            connection.needs_rollback = True
            connection.rollback_exc = exc
        raise


def on_commit(func, using=None, robust=False):
    """
    Register `func` to be called when the current transaction is committed.
    If the current transaction is rolled back, `func` will not be called.
    """
    connection = get_connection(using)
    if _uses_async_transaction(connection):
        return connection.on_async_commit(func, robust)
    return connection.on_commit(func, robust)


#################################
# Decorators / context managers #
#################################


class Atomic(ContextDecorator):
    """
    Guarantee the atomic execution of a given block.

    An instance can be used either as a decorator or as a context manager.

    When it's used as a decorator, __call__ wraps the execution of the
    decorated function in the instance itself, used as a context manager.

    When it's used as a context manager, __enter__ creates a transaction or a
    savepoint, depending on whether a transaction is already in progress, and
    __exit__ commits the transaction or releases the savepoint on normal exit,
    and rolls back the transaction or to the savepoint on exceptions.

    It's possible to disable the creation of savepoints if the goal is to
    ensure that some code runs within a transaction without creating overhead.

    A stack of savepoint identifiers is maintained as an attribute of the
    connection. None denotes the absence of a savepoint.

    This allows reentrancy even if the same AtomicWrapper is reused. For
    example, it's possible to define `oa = atomic('other')` and use `@oa` or
    `with oa:` multiple times.

    Since database connections are thread-local, this is thread-safe.

    An atomic block can be tagged as durable. In this case, a RuntimeError is
    raised if it's nested within another atomic block. This guarantees
    that database changes in a durable block are committed to the database when
    the block exits without error.

    This is a private API.
    """

    def __init__(
        self,
        using,
        savepoint,
        durable,
        isolation_level=None,
        read_only=None,
        deferrable=None,
    ):
        self.using = using
        self.savepoint = savepoint
        self.durable = durable
        self.isolation_level = isolation_level
        self.read_only = read_only
        self.deferrable = deferrable
        self._from_testcase = False
        self._async_contexts = ContextVar(
            f"django_async_atomic_contexts_{id(self)}", default=()
        )

    def __call__(self, func):
        if iscoroutinefunction(func):

            @wraps(func)
            async def inner(*args, **kwargs):
                async with self._recreate_cm():
                    return await func(*args, **kwargs)

            return inner
        return super().__call__(func)

    def __enter__(self):
        if any(
            option is not None
            for option in (self.isolation_level, self.read_only, self.deferrable)
        ):
            raise NotSupportedError(
                "Transaction options require a native async atomic block."
            )
        connection = get_connection(self.using)

        if (
            self.durable
            and connection.atomic_blocks
            and not connection.atomic_blocks[-1]._from_testcase
        ):
            raise RuntimeError(
                "A durable atomic block cannot be nested within another "
                "atomic block."
            )
        if not connection.in_atomic_block:
            # Reset state when entering an outermost atomic block.
            connection.commit_on_exit = True
            connection.needs_rollback = False
            if not connection.get_autocommit():
                # Pretend we're already in an atomic block to bypass the code
                # that disables autocommit to enter a transaction, and make a
                # note to deal with this case in __exit__.
                connection.in_atomic_block = True
                connection.commit_on_exit = False

        if connection.in_atomic_block:
            # We're already in a transaction; create a savepoint, unless we
            # were told not to or we're already waiting for a rollback. The
            # second condition avoids creating useless savepoints and prevents
            # overwriting needs_rollback until the rollback is performed.
            if self.savepoint and not connection.needs_rollback:
                sid = connection.savepoint()
                connection.savepoint_ids.append(sid)
            else:
                connection.savepoint_ids.append(None)
        else:
            connection.set_autocommit(
                False, force_begin_transaction_with_broken_autocommit=True
            )
            connection.in_atomic_block = True

        if connection.in_atomic_block:
            connection.atomic_blocks.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        connection = get_connection(self.using)

        if connection.in_atomic_block:
            connection.atomic_blocks.pop()

        if connection.savepoint_ids:
            sid = connection.savepoint_ids.pop()
        else:
            # Prematurely unset this flag to allow using commit or rollback.
            connection.in_atomic_block = False

        try:
            if connection.closed_in_transaction:
                # The database will perform a rollback by itself.
                # Wait until we exit the outermost block.
                pass

            elif exc_type is None and not connection.needs_rollback:
                if connection.in_atomic_block:
                    # Release savepoint if there is one
                    if sid is not None:
                        try:
                            connection.savepoint_commit(sid)
                        except DatabaseError:
                            try:
                                connection.savepoint_rollback(sid)
                                # The savepoint won't be reused. Release it to
                                # minimize overhead for the database server.
                                connection.savepoint_commit(sid)
                            except Error:
                                # If rolling back to a savepoint fails, mark
                                # for rollback at a higher level and avoid
                                # shadowing the original exception.
                                connection.needs_rollback = True
                            raise
                else:
                    # Commit transaction
                    try:
                        connection.commit()
                    except DatabaseError:
                        try:
                            connection.rollback()
                        except Error:
                            # An error during rollback means that something
                            # went wrong with the connection. Drop it.
                            connection.close()
                        raise
            else:
                # This flag will be set to True again if there isn't a
                # savepoint allowing to perform the rollback at this level.
                connection.needs_rollback = False
                if connection.in_atomic_block:
                    # Roll back to savepoint if there is one, mark for rollback
                    # otherwise.
                    if sid is None:
                        connection.needs_rollback = True
                    else:
                        try:
                            connection.savepoint_rollback(sid)
                            # The savepoint won't be reused. Release it to
                            # minimize overhead for the database server.
                            connection.savepoint_commit(sid)
                        except Error:
                            # If rolling back to a savepoint fails, mark for
                            # rollback at a higher level and avoid shadowing
                            # the original exception.
                            connection.needs_rollback = True
                else:
                    # Roll back transaction
                    try:
                        connection.rollback()
                    except Error:
                        # An error during rollback means that something
                        # went wrong with the connection. Drop it.
                        connection.close()

        finally:
            # Outermost block exit when autocommit was enabled.
            if not connection.in_atomic_block:
                if connection.closed_in_transaction:
                    connection.connection = None
                else:
                    connection.set_autocommit(True)
            # Outermost block exit when autocommit was disabled.
            elif not connection.savepoint_ids and not connection.commit_on_exit:
                if connection.closed_in_transaction:
                    connection.connection = None
                else:
                    connection.in_atomic_block = False

    async def __aenter__(self):
        connection = get_connection(self.using)
        if not connection.features.supports_async:
            raise NotSupportedError(
                "This database backend does not support native async transactions."
            )

        blocks_by_alias = _async_atomic_blocks.get()
        blocks = blocks_by_alias.get(connection.alias, ())
        if self.durable and blocks and not blocks[-1]._from_testcase:
            raise RuntimeError(
                "A durable atomic block cannot be nested within another atomic block."
            )

        transaction_options = {}
        if not self.savepoint:
            transaction_options["savepoint"] = False
        if self.isolation_level is not None:
            transaction_options["isolation_level"] = self.isolation_level
        if self.read_only is not None:
            transaction_options["read_only"] = self.read_only
        if self.deferrable is not None:
            transaction_options["deferrable"] = self.deferrable

        context = connection.atransaction(**transaction_options)
        await context.__aenter__()

        updated_blocks = blocks_by_alias.copy()
        updated_blocks[connection.alias] = (*blocks, self)
        blocks_token = _async_atomic_blocks.set(updated_blocks)
        contexts = self._async_contexts.get()
        self._async_contexts.set((*contexts, (context, blocks_token)))

    async def __aexit__(self, exc_type, exc_value, traceback):
        contexts = self._async_contexts.get()
        if not contexts:
            raise RuntimeError("Async atomic block is not active.")

        context, blocks_token = contexts[-1]
        connection = get_connection(self.using)
        blocks = _async_atomic_blocks.get().get(connection.alias, ())
        completed = False
        try:
            result = await context.__aexit__(exc_type, exc_value, traceback)
            completed = True
        finally:
            self._async_contexts.set(contexts[:-1])
            _async_atomic_blocks.reset(blocks_token)
        if completed and len(blocks) == 1:
            connection.run_async_commit_hooks()
        return result


def atomic(
    using=None,
    savepoint=True,
    durable=False,
    *,
    isolation_level=None,
    read_only=None,
    deferrable=None,
):
    # Bare decorator: @atomic -- although the first argument is called
    # `using`, it's actually the function being decorated.
    if callable(using):
        return Atomic(
            DEFAULT_DB_ALIAS,
            savepoint,
            durable,
            isolation_level,
            read_only,
            deferrable,
        )(using)
    # Decorator: @atomic(...) or context manager: with atomic(...): ...
    else:
        return Atomic(
            using,
            savepoint,
            durable,
            isolation_level,
            read_only,
            deferrable,
        )


def _non_atomic_requests(view, using):
    try:
        databases = view._non_atomic_requests | {using}
    except AttributeError:
        databases = {using}

    @wraps(view)
    def wrapper(*args, **kwargs):
        return view(*args, **kwargs)

    wrapper._non_atomic_requests = databases
    return wrapper


def non_atomic_requests(using=None):
    if callable(using):
        return _non_atomic_requests(using, DEFAULT_DB_ALIAS)
    else:
        if using is None:
            using = DEFAULT_DB_ALIAS
        return lambda view: _non_atomic_requests(view, using)
