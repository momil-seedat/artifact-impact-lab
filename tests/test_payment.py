"""Tests for the payment service.

``test_process_payment`` exercises ``processPayment`` and is the Test -> Service
(T1 -> St12) edge in the artifact impact graph.
"""

from services.payment_service import processPayment, loadConfig


def test_process_payment():
    config = loadConfig()
    receipt = processPayment(100, config=config)
    assert receipt["status"] == "ok"
    assert receipt["amount"] == 100


def test_process_payment_uses_timeout():
    config = loadConfig()
    receipt = processPayment(5, config=config)
    assert receipt["timeout"] == int(config["payment.timeout"])
