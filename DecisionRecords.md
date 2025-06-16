# ADR-001: Deploy Verify Service as Independent Microservice with Dedicated Database

## Status
Accepted (2025-05-29)

## Context
The `verify` service handles image and video analysis for our tweet fact-checking system. As part of our transition from a monolith to microservices, we need this service to be independently deployable and scalable.

Key goals:
- Decoupled deployment and scaling of the `verify` service.
- Isolation of image/video processing workloads.
- Decoupled database schema for focused data modeling and access control.
- Seamless integration with the rest of the architecture.

## Decision
We will develop and deploy the `verify` service as an independent microservice, with its own dedicated database schema.

## Consequences

### Positive
- Independent deployment and CI/CD pipelines.
- Scales separately from the rest of the system.
- Decouples media processing concerns from other services.
- Clear service boundaries and ownership.
- Easier to maintain, test, and monitor in isolation.

### Negative
- Adds operational overhead (additional services to deploy, monitor).
- Requires inter-service communication for example for receiving tweet context.
- Schema/data duplication risk if not coordinated properly.

### Neutral
- Needs thoughtful observability strategy (logs, metrics, tracing).
- Database migrations and schema management are scoped only to `verify`.

## Alternatives Considered

### Shared Database with Monolith
- **Pros**: Easy to implement short-term.
- **Cons**: Reduces isolation, creates strong coupling, makes scaling harder.



