from unittest import mock

from django.core.exceptions import EmptyResultSet
from django.db import DEFAULT_DB_ALIAS, DatabaseError, connection
from django.db.models.sql import Query
from django.db.models.sql.compiler import SQLCompiler
from django.db.models.sql.constants import MULTI, SINGLE
from django.test import TestCase
from django.utils.deprecation import RemovedInDjango2028Warning

from .models import Item


class SQLCompilerTest(TestCase):
    def test_repr(self):
        query = Query(Item)
        compiler = query.get_compiler(DEFAULT_DB_ALIAS, connection)
        self.assertEqual(
            repr(compiler),
            f"<SQLCompiler model=Item connection="
            f"<DatabaseWrapper vendor={connection.vendor!r} alias='default'> "
            f"using='default'>",
        )

    def test_execute_sql_suppresses_cursor_closing_failure_on_exception(self):
        query = Query(Item)
        compiler = SQLCompiler(query, connection, None)
        cursor = mock.MagicMock()

        # When execution fails, the cursor may have been closed.
        # Django's attempt to close it again will fail, and needs catching.
        execute_err = DatabaseError("execute failed")
        cursor.execute.side_effect = execute_err
        cursor.close.side_effect = DatabaseError("close failed")

        with mock.patch.object(connection, "cursor", return_value=cursor):
            with self.assertRaises(DatabaseError) as ctx:
                compiler.execute_sql("SELECT 1", [])

        # There is no irrelevant context from trying to close a closed cursor.
        exc = ctx.exception
        self.assertIs(exc, execute_err)
        self.assertIsNone(exc.__cause__)
        self.assertTrue(exc.__suppress_context__)

    async def test_aexecute_sql_single_uses_async_cursor(self):
        query = Query(Item).exists()
        compiler = query.get_compiler(DEFAULT_DB_ALIAS, connection)
        sql, params = compiler.as_sql()
        cursor = mock.MagicMock()
        cursor.aexecute = mock.AsyncMock()
        cursor.afetchone = mock.AsyncMock(return_value=(1, "extra column"))
        async_cursor = mock.MagicMock()
        async_cursor.__aenter__.return_value = cursor

        with mock.patch.object(
            connection,
            "acursor",
            return_value=async_cursor,
            create=True,
        ) as acursor:
            result = await compiler.aexecute_sql(SINGLE)

        self.assertEqual(result, (1,))
        acursor.assert_called_once_with()
        cursor.aexecute.assert_awaited_once_with(sql, params)
        cursor.afetchone.assert_awaited_once_with()
        async_cursor.__aexit__.assert_awaited_once()

    async def test_aexecute_sql_single_empty_result_set(self):
        compiler = SQLCompiler(Query(Item), connection, DEFAULT_DB_ALIAS)

        with (
            mock.patch.object(compiler, "as_sql", side_effect=EmptyResultSet),
            mock.patch.object(connection, "acursor", create=True) as acursor,
        ):
            result = await compiler.aexecute_sql(SINGLE)

        self.assertIsNone(result)
        acursor.assert_not_called()

    async def test_aexecute_sql_single_no_row(self):
        compiler = (
            Query(Item)
            .exists()
            .get_compiler(
                DEFAULT_DB_ALIAS,
                connection,
            )
        )
        cursor = mock.MagicMock()
        cursor.aexecute = mock.AsyncMock()
        cursor.afetchone = mock.AsyncMock(return_value=None)
        async_cursor = mock.MagicMock()
        async_cursor.__aenter__.return_value = cursor

        with mock.patch.object(
            connection,
            "acursor",
            return_value=async_cursor,
            create=True,
        ):
            result = await compiler.aexecute_sql(SINGLE)

        self.assertIsNone(result)

    async def test_aexecute_sql_closes_cursor_context_after_error(self):
        compiler = (
            Query(Item)
            .exists()
            .get_compiler(
                DEFAULT_DB_ALIAS,
                connection,
            )
        )
        cursor = mock.MagicMock()
        cursor.aexecute = mock.AsyncMock(side_effect=DatabaseError("execute failed"))
        cursor.afetchone = mock.AsyncMock()
        async_cursor = mock.MagicMock()
        async_cursor.__aenter__.return_value = cursor

        with (
            mock.patch.object(
                connection,
                "acursor",
                return_value=async_cursor,
                create=True,
            ),
            self.assertRaisesMessage(DatabaseError, "execute failed"),
        ):
            await compiler.aexecute_sql(SINGLE)

        cursor.afetchone.assert_not_awaited()
        async_cursor.__aexit__.assert_awaited_once()

    async def test_aexecute_sql_multi_fetches_all_rows(self):
        compiler = SQLCompiler(Query(Item), connection, DEFAULT_DB_ALIAS)
        sql, params = compiler.as_sql()
        cursor = mock.MagicMock()
        cursor.aexecute = mock.AsyncMock()
        cursor.afetchall = mock.AsyncMock(return_value=[(1,), (2,)])
        async_cursor = mock.MagicMock()
        async_cursor.__aenter__.return_value = cursor

        with mock.patch.object(
            connection,
            "acursor",
            return_value=async_cursor,
            create=True,
        ):
            results = await compiler.aexecute_sql(MULTI)
            batches = [batch async for batch in results]

        self.assertEqual(batches, [[(1,), (2,)]])
        cursor.aexecute.assert_awaited_once_with(sql, params)
        cursor.afetchall.assert_awaited_once_with()
        async_cursor.__aexit__.assert_awaited_once()

    async def test_aexecute_sql_multi_fetches_rows_in_chunks(self):
        compiler = SQLCompiler(Query(Item), connection, DEFAULT_DB_ALIAS)
        cursor = mock.MagicMock()
        cursor.aexecute = mock.AsyncMock()
        cursor.afetchmany = mock.AsyncMock(
            side_effect=[[(1,), (2,)], [(3,)], []],
        )
        async_cursor = mock.MagicMock()
        async_cursor.__aenter__.return_value = cursor

        with mock.patch.object(
            connection,
            "acursor",
            return_value=async_cursor,
            create=True,
        ):
            results = await compiler.aexecute_sql(
                MULTI,
                chunked_fetch=True,
                chunk_size=2,
            )
            batches = [batch async for batch in results]

        self.assertEqual(batches, [[(1,), (2,)], [(3,)]])
        self.assertEqual(
            cursor.afetchmany.await_args_list,
            [mock.call(2), mock.call(2), mock.call(2)],
        )
        async_cursor.__aexit__.assert_awaited_once()

    async def test_aexecute_sql_multi_empty_result_set(self):
        compiler = SQLCompiler(Query(Item), connection, DEFAULT_DB_ALIAS)

        with (
            mock.patch.object(compiler, "as_sql", side_effect=EmptyResultSet),
            mock.patch.object(connection, "acursor", create=True) as acursor,
        ):
            results = await compiler.aexecute_sql(MULTI)
            batches = [batch async for batch in results]

        self.assertEqual(batches, [])
        acursor.assert_not_called()

    async def test_aexecute_sql_multi_closes_partially_consumed_results(self):
        compiler = SQLCompiler(Query(Item), connection, DEFAULT_DB_ALIAS)
        cursor = mock.MagicMock()
        cursor.aexecute = mock.AsyncMock()
        cursor.afetchmany = mock.AsyncMock(return_value=[(1,)])
        async_cursor = mock.MagicMock()
        async_cursor.__aenter__.return_value = cursor

        with mock.patch.object(
            connection,
            "acursor",
            return_value=async_cursor,
            create=True,
        ):
            results = await compiler.aexecute_sql(
                MULTI,
                chunked_fetch=True,
                chunk_size=1,
            )
            self.assertEqual(await anext(results), [(1,)])
            await results.aclose()

        async_cursor.__aexit__.assert_awaited_once()

    async def test_aresults_iter_flattens_batches_and_converts_rows(self):
        compiler = (
            Query(Item)
            .exists()
            .get_compiler(
                DEFAULT_DB_ALIAS,
                connection,
            )
        )
        compiler.as_sql()
        expression = compiler.select[0][0]

        def add_one(value, expression, connection):
            return value + 1

        async def results():
            yield [(1,), (2,)]
            yield [(3,)]

        with (
            mock.patch.object(
                compiler,
                "aexecute_sql",
                new=mock.AsyncMock(return_value=results()),
            ) as aexecute_sql,
            mock.patch.object(
                compiler,
                "get_converters",
                return_value={0: ([add_one], expression)},
            ),
        ):
            rows = [
                row
                async for row in compiler.aresults_iter(
                    tuple_expected=True,
                    chunked_fetch=True,
                    chunk_size=2,
                )
            ]

        self.assertEqual(rows, [(2,), (3,), (4,)])
        aexecute_sql.assert_awaited_once_with(
            MULTI,
            chunked_fetch=True,
            chunk_size=2,
        )

    async def test_aresults_iter_closes_partially_consumed_results(self):
        compiler = (
            Query(Item)
            .exists()
            .get_compiler(
                DEFAULT_DB_ALIAS,
                connection,
            )
        )
        compiler.as_sql()
        results_closed = False

        async def results():
            nonlocal results_closed
            try:
                yield [(1,)]
                yield [(2,)]
            finally:
                results_closed = True

        with mock.patch.object(
            compiler,
            "aexecute_sql",
            new=mock.AsyncMock(return_value=results()),
        ):
            rows = compiler.aresults_iter()
            self.assertEqual(await anext(rows), [1])
            await rows.aclose()

        self.assertIs(results_closed, True)

    async def test_ahas_results_uses_async_execution(self):
        compiler = SQLCompiler(Query(Item), connection, DEFAULT_DB_ALIAS)

        with mock.patch.object(
            compiler,
            "aexecute_sql",
            new=mock.AsyncMock(return_value=(1,)),
        ) as aexecute_sql:
            result = await compiler.ahas_results()

        self.assertIs(result, True)
        aexecute_sql.assert_awaited_once_with(SINGLE)

    async def test_ahas_results_preserves_custom_has_results_override(self):
        class CustomCompiler(SQLCompiler):
            has_results_called = False

            def has_results(self):
                self.has_results_called = True
                return True

        query = Query(Item)
        compiler = CustomCompiler(query, connection, DEFAULT_DB_ALIAS)

        result = await compiler.ahas_results()

        self.assertIs(result, True)
        self.assertIs(compiler.has_results_called, True)

    # RemovedInDjango2028Warning: When the deprecation ends, remove this
    # test.
    def test_quote_name_unless_alias_deprecation(self):
        query = Query(Item)
        compiler = SQLCompiler(query, connection, None)
        msg = (
            "SQLCompiler.quote_name_unless_alias() is deprecated. "
            "Use .quote_name() instead."
        )
        with self.assertWarnsMessage(RemovedInDjango2028Warning, msg) as ctx:
            self.assertEqual(
                compiler.quote_name_unless_alias("name"),
                compiler.quote_name("name"),
            )
        self.assertEqual(ctx.filename, __file__)
