"""Order service.

Provides :func:`completeOrder`, which calls
:func:`payment_service.processPayment`. That call is the Service -> Service
(St22 -> St12) edge in the artifact impact graph.
"""

from services.payment_service import processPayment


def completeOrder(order):
    """Complete an order by charging its total via the payment service."""
    amount = order["total"]
    receipt = processPayment(amount)
    return {
        "order_id": order["id"],
        "receipt": receipt,
        "status": "completed",
    }
