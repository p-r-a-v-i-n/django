from django.test import SimpleTestCase, override_settings
from django.forms import Form
from django.template import Context, Template
from django.utils.csp import CSP

class FormWithJsMedia(Form):
    class Media:
        js = ["path/to/js_file.js"]

@override_settings(
    STATIC_URL="/static/",
    MIDDLEWARE=[
        "django.middleware.security.SecurityMiddleware",
        "django.middleware.csp.ContentSecurityPolicyMiddleware",
    ],
    TEMPLATES=[{
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.template.context_processors.csp",
            ],
        },
    }],
    SECURE_CSP={
        "default-src": [CSP.SELF],
        "script-src": [CSP.SELF, CSP.NONCE],
    }
)
class CSPMediaTest(SimpleTestCase):
    def test_form_media_js_missing_nonce(self):
        form = FormWithJsMedia()
        tpl = Template("{% load static %}{{ form.media }}")
        rendered = tpl.render(Context({"form": form}))
        self.assertIn('<script src="/static/path/to/js_file.js">', rendered)
        self.assertIn('nonce="', rendered)
