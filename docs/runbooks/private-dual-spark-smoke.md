# Private dual-Spark DeepSeek/Hermes smoke runbook

This runbook operates the temporary `DeepSeek-V4-Flash-0731` evaluation on
`spark-api` and `spark-lab`. It is not a production service. Nothing here
installs a restart policy, public tunnel, messaging channel, or Hermes gateway.

## Safety boundary

- Qwen is stopped before DeepSeek starts. It is never a rollback target and no
  command in this workflow starts it again.
- The existing LiteLLM, Postgres, Cloudflared, default Hermes, and Hermesia
  services are protected state. A changed identity/hash fails acceptance.
- Secrets live only in operator-created mode-0600 files. Do not paste KVM,
  Spark login, origin, LiteLLM master, database, or virtual-key values into a
  command line, repository file, report, or chat.
- The CX-7 apply and OpenMPI package installation need an interactive `sudo`
  session on each Spark. The scripts do not store or pipe sudo passwords.
- Run the Hermes suite on the Mac mini. Its temporary `HERMES_HOME`, Docker
  config, Colima profile, containers, and fake failure providers are deleted at
  the end of each run. Existing Colima profiles are not selected or modified.

## One-time prerequisites

On both Sparks, verify passwordless SSH in the required direction and install
the MPI launcher packages interactively if they are absent:

```bash
ssh spark-api 'command -v mpirun'
ssh spark-lab 'command -v orted || command -v prte'
```

The head must trust and reach the worker by the exact `WORKER_HOST` stored in
the untracked `.env.dspark`. Do not disable host-key checking; add the verified
worker host key and test `ssh -o BatchMode=yes "$WORKER_HOST" true`.

The Mac mini needs Hermes v0.19.0 or the explicitly requalified successor,
Colima, and the Docker CLI. Docker Desktop is not required. Smoke runs use a
unique `dspark-hermes-smoke-*` Colima profile with runtime Docker.

## Prepare and verify the direct fabric

On `spark-api`, apply the head profile:

```bash
deployments/private-smoke/scripts/apply-cx7-network.sh --check --role head
deployments/private-smoke/scripts/apply-cx7-network.sh --apply --role head
```

On `spark-lab`, apply the worker profile:

```bash
deployments/private-smoke/scripts/apply-cx7-network.sh --check --role worker
deployments/private-smoke/scripts/apply-cx7-network.sh --apply --role worker
```

Then verify from `spark-api` with the same worker target used by `.env.dspark`:

```bash
HEAD_HOST=localhost WORKER_HOST="$WORKER_HOST" \
  deployments/private-smoke/scripts/verify-fabric.sh --require-persistent
deployments/private-smoke/scripts/deploy-dspark.sh --prepare-only
```

The apply path backs up the existing netplan, uses a timed rollback, and must
leave the default route and Tailscale route unchanged. Do not continue unless
jumbo ping, bidirectional RDMA bandwidth, NCCL, persistent addresses, and the
route-preservation checks all pass.

## Start and pass the direct gate

```bash
deployments/private-smoke/scripts/deploy-dspark.sh --direct-gate
```

This inventories Qwen, requires the typed stop confirmation, leaves Qwen
stopped, starts worker then head, runs the direct semantic suite and benchmark,
and creates a private `artifacts/acceptance/<timestamp>/` directory. A failure
stops both DeepSeek ranks and does not start Qwen.

Status and manual stop commands:

```bash
./status-deepseek-v4-flash-dspark.sh --expect running
./stop-deepseek-v4-flash-dspark.sh
./status-deepseek-v4-flash-dspark.sh --expect stopped
```

## Start the private LiteLLM gate

Create the untracked LiteLLM secret files and `.env` described by
`deployments/private-smoke/litellm/.env.example`, all mode 0600. Then:

```bash
deployments/private-smoke/litellm/deploy.sh
deployments/private-smoke/litellm/smoke.sh --all-interfaces
```

The only published listener is the head Tailscale address on port 4001. The
model-scoped virtual key cannot create keys, read configuration, call another
model, access the Docker socket, reach the public internet, or reach the
existing gateway. The vLLM origin remains bridge-only.

To stop only the private gateway:

```bash
docker compose -p dspark-private-litellm \
  --env-file deployments/private-smoke/litellm/.env \
  -f deployments/private-smoke/litellm/docker-compose.yml down --remove-orphans
docker volume rm dspark-private-litellm-prisma-cache
deployments/private-smoke/litellm/egress-policy.sh --remove
```

## Run Hermes twice on the Mac mini

Copy this repository revision to a private path on the Mac mini. Transfer the
virtual inference key with an encrypted channel into a new mode-0600 file; do
not pass the key value as an argument. Point the output at the matching Spark
acceptance run's `hermes/` evidence directory (or securely copy the two result
JSON files there afterward):

```bash
HERMES_SMOKE_BASE_URL=http://TAILSCALE_HEAD_ADDRESS:4001/v1 \
HERMES_SMOKE_KEY_FILE=/private/mode-0600/hermes-inference.key \
HERMES_SMOKE_OUTPUT_DIR=/private/evidence/hermes \
  deployments/private-smoke/hermes/run-suite.sh --repeat 2
```

The URL placeholder must be replaced locally; never write the private address
into committed files or accepted evidence. The suite performs only synthetic
create/read/transform work. It also proves no host mounts, no terminal network,
no skills/MCP/memory/gateway inheritance, one-attempt failures, and unchanged
default/Hermesia configuration.

## Run full acceptance

Place exactly two Hermes result JSON files in the run directory's `hermes/`
folder. On `spark-api`:

```bash
deployments/private-smoke/run-acceptance.sh --validate-fixtures
deployments/private-smoke/run-acceptance.sh --live \
  --run-dir artifacts/acceptance/<timestamp> \
  --hermes-results artifacts/acceptance/<timestamp>/hermes
```

Live acceptance reruns the LiteLLM performance workload, checks exact LiteLLM
spend-log and vLLM completion deltas, then runs a full 1,800-second C1 soak with
five-second node/queue/memory/restart/preemption samples. It writes only a
schema-valid, sanitized `accepted.json` containing allowlisted aggregates and a
hash chain. Raw responses, keys, private addresses, and host paths are not
accepted evidence.

Any failure writes `rejected.json`, stops both DeepSeek ranks, removes only the
private LiteLLM stack and egress rules, and leaves Qwen stopped.

## Purge Qwen only after acceptance

First run the nondestructive contract check:

```bash
deployments/private-smoke/scripts/purge-qwen.sh --verify-only
```

Then, from an interactive terminal on `spark-api`, bind the exact manifest and
accepted report:

```bash
deployments/private-smoke/scripts/purge-qwen.sh \
  --manifest artifacts/acceptance/<timestamp>/qwen-manifest.json \
  --gate-report artifacts/acceptance/<timestamp>/accepted.json
```

The purge first re-proves both DeepSeek ranks and the private LiteLLM gateway.
Read the resolved run ID and manifest SHA-256, then type both requested
confirmations exactly. The first moves only the inode-verified allowlisted
targets into same-filesystem quarantine and removes the exact container/image.
The second permanently deletes that quarantine. If the second confirmation is
not entered, the quarantine remains recoverable and the command fails. Any
recorded or current Qwen supervisor entry blocks retirement.

After purge, confirm the Qwen container/image/model/service targets remain
absent and the existing public gateway containers retain the pre-run identity.
DeepSeek and the private LiteLLM may remain manually running for inspection;
they have no automatic restart behavior.
