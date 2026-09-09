# Payment Requirements

This document captures the requirements for the payment subsystem.

Each requirement carries a machine-readable trace annotation so that
`extract_artifacts.py` can link requirements (R) to the configuration
properties (C) they constrain.

Annotation format (parsed by the extractor):

    <!-- @id: R1 -->
    <!-- @config: payment.timeout -->

---

## R1 — Payment must time out

<!-- @id: R1 -->
<!-- @config: payment.timeout -->

A payment attempt MUST abort if the payment gateway does not respond within a
configurable timeout. The timeout is controlled by the `payment.timeout`
configuration property and is consumed by the payment service when it processes
a payment.

## R2 — Payments must be retried

<!-- @id: R2 -->
<!-- @config: payment.retry.count -->

A failed payment MUST be retried up to a configurable number of times, governed
by the `payment.retry.count` configuration property.

## R3 — Payments target a configured gateway

<!-- @id: R3 -->
<!-- @config: payment.gateway.url -->

All payments MUST be sent to the gateway identified by the
`payment.gateway.url` configuration property.
