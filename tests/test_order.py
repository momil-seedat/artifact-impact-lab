"""Tests for the order service.

``test_complete_order`` exercises ``completeOrder``, which transitively reaches
``processPayment``.
"""

from services.order_service import completeOrder


def test_complete_order():
    order = {"id": "A-1", "total": 250}
    result = completeOrder(order)
    assert result["status"] == "completed"
    assert result["receipt"]["amount"] == 250
