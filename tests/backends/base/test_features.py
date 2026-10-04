from django.db import connection
from django.test import SimpleTestCase


class TestDatabaseFeatures(SimpleTestCase):
    def test_async_not_supported_by_default(self):
        self.assertIs(connection.features.supports_async, False)

    def test_nonexistent_feature(self):
        self.assertFalse(hasattr(connection.features, "nonexistent"))
