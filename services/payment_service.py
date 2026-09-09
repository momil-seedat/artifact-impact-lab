"""Payment service.

Provides :func:`processPayment`, which reads the ``payment.timeout`` (and other)
configuration properties. The literal config keys used here are what the
extractor links back to the Config (C) artifacts and, transitively, to the
Requirement (R) artifacts.
"""

import os


def loadConfig(path=None):
    """Parse a Java-style ``.properties`` file into a dict."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "payment.properties",
        )
    config = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config


def processPayment(amount, config=None):
    """Process a single payment.

    Consumes the ``payment.timeout``, ``payment.retry.count`` and
    ``payment.gateway.url`` configuration properties.
    """
    if config is None:
        config = loadConfig()

    timeout = int(config["payment.timeout"])
    retries = int(config["payment.retry.count"])
    gateway = config["payment.gateway.url"]

    last_error = None
    for attempt in range(retries):
        try:
            return _callGateway(gateway, amount, timeout)
        except TimeoutError as exc:  # pragma: no cover - illustrative
            last_error = exc
    raise RuntimeError("payment failed after %d attempts" % retries) from last_error


def _callGateway(gateway, amount, timeout):
    """Pretend to call the payment gateway. Returns a receipt dict."""
    return {
        "gateway": gateway,
        "amount": amount,
        "timeout": timeout,
        "status": "ok",
    }
