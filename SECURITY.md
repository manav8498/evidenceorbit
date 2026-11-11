# Security policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting

Please report vulnerabilities privately. Open a GitHub security advisory, or email the maintainers listed in the repository, with:

- affected version
- a trajectory or unit test that demonstrates the issue
- whether the issue is in detection (false negative on a safety property), recovery (unwanted rewind or side-effect), or the MCP/CLI surface

Do not open a public issue for an unfixed vulnerability.

## Recovery safety

`apply_recovery` is gated. The default policy is propose-only. External or destructive tool effects inside the rewind window are refused unless `RecoveryPolicy.allow_destructive` is set explicitly. EvidenceOrbit does not execute agent tools itself. Adapters must not map a recovery proposal onto a live side-effecting tool without the same gate.

## Audit integrity

Audit logs are hash-chained. If `AuditLog` fails verification on load, the process raises. Do not delete rows to "clean up" a log that will be used as an evidence record.
