# Security Policy

## Overview

Pinak takes security seriously. This document outlines our security practices, policies, and procedures for reporting vulnerabilities.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Features

### Authentication & Authorization

- **JWT-based Authentication**: All API endpoints require valid JWT bearer tokens
- **Role-Based Access Control (RBAC)**: Four role levels (admin, user, guest, service) with granular permissions
- **Multi-tenant Isolation**: File-system level segregation by tenant and project ID
- **Token Validation**: Mandatory tenant and project_id claims in all tokens

### Data Protection

- **Local-First Architecture**: All data stored locally by default with optional sync
- **Tamper-Evident Audit Logs**: Hash-chained event logs with verification utilities
- **Privacy Controls**: Configurable redaction rules for sensitive data
- **Encryption**: Supports encrypted storage backends (configure via environment)

### Security Best Practices

1. **Environment Variables**: Never commit secrets to version control
   - `PINAK_JWT_SECRET`: Must be set for authentication
   - Use strong, randomly generated secrets in production

2. **Token Expiration**: Set appropriate token expiration times
   - Development: 1-8 hours
   - Production: 15-60 minutes with refresh token rotation

3. **Network Security**: 
   - Use TLS/HTTPS in production
   - Implement rate limiting at the API gateway level
   - Configure CORS policies appropriately

4. **Dependency Management**:
   - Regular security audits via `pip-audit`
   - Automated vulnerability scanning in CI/CD
   - Keep dependencies up to date

## Reporting a Vulnerability

We appreciate responsible disclosure of security vulnerabilities.

### How to Report

1. **Email**: Send details to **security@pinak-setu.com**
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)
   - Your contact information

3. **Do NOT**:
   - Publicly disclose the vulnerability before we've addressed it
   - Test against production systems without permission
   - Access or modify user data

### What to Expect

- **Initial Response**: Within 48 hours of submission
- **Confirmation**: Within 5 business days
- **Regular Updates**: Every 7 days until resolution
- **Disclosure Timeline**: 90 days or coordinated disclosure date
- **Credit**: Public acknowledgment in release notes (if desired)

### Severity Classification

| Severity | Description | Response Time |
|----------|-------------|---------------|
| Critical | Remote code execution, authentication bypass | 24-48 hours |
| High | Privilege escalation, data breach | 3-7 days |
| Medium | Information disclosure, DoS | 14 days |
| Low | Minor information leaks | 30 days |

## Security Audit Trail

All security-relevant events are logged in the audit trail:

- Authentication attempts (success/failure)
- Authorization decisions
- Data access (read/write/delete)
- Configuration changes
- Administrative actions

Audit logs are tamper-evident using hash chaining and can be verified using built-in utilities.

## Compliance

Pinak is designed with the following compliance frameworks in mind:

- **GDPR**: Data minimization, right to erasure, data portability
- **SOC 2**: Security, availability, confidentiality controls
- **CCPA**: Privacy rights and data handling
- **ISO 27001**: Information security management

For compliance documentation and certifications, contact: compliance@pinak-setu.com

## Security Testing

We perform regular security assessments:

- **Automated Scanning**: Every commit via GitHub Actions
- **Dependency Audits**: Weekly via pip-audit and Trivy
- **Penetration Testing**: Annually (production systems)
- **Code Reviews**: All security-related changes

## Known Issues

Current known security considerations:

1. **Development Mode**: Default JWT secret should never be used in production
2. **File System Storage**: Ensure proper file permissions on data directories
3. **Redis Cache**: Configure authentication if Redis is used in production

## Additional Resources

- [Contributing Guidelines](CONTRIBUTING.md)
- [API Documentation](docs/)
- [Remediation & Hardening Plan](docs/remediation_plan.md)
- [Enterprise Reference Architecture](pinak_enterprise_reference.md)

## Contact

- **Security Issues**: security@pinak-setu.com
- **General Support**: support@pinak-setu.com
- **GitHub Issues**: https://github.com/Pinak-Setu/Pinak_Projects/issues

---

**Last Updated**: 2025-01-11  
**Version**: 1.0
