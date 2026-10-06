# Failure Runbook

## 1. Classification workflow failure

### Symptoms
- `/classify` does not complete successfully.
- The response indicates that the classification workflow is not available or could not complete.

### Checks
- Check the API service logs.
- Check whether the classification graph and its agents are available.
- Check whether the retrieval and verification components are responding.

### Recovery
- Restart the API service if the application process is unavailable.
- Re-run a classification request after the service is available.
- If the classification workflow remains unavailable, keep the request in specialist review rather than treating it as a successful automated classification.

## 2. Rate-limit responses

### Symptoms
- `/classify` returns HTTP `429`.
- The response indicates that the rate limit has been exceeded.

### Checks
- Confirm that the client is sending more than the configured limit of 10 requests per 60 seconds.
- Check whether repeated requests are coming from the same client.

### Recovery
- Wait for the rate-limit window to expire before retrying.
- Reduce the request rate.
- Do not increase the production rate limit solely to bypass a client-side rate-limit failure.