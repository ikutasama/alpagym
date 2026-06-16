# NCCL Transport

The NCCL transport moves completed `EpisodeOutput` payloads from rollout workers to policy workers
in a Cosmos-RL job. TCPStore carries the small control plane. NCCL carries the tensor payloads.
Redis carries Cosmos discard-cleanup messages.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Runtime roles](#runtime-roles)
- [Modules](#modules)
- [End-to-end flow](#end-to-end-flow)
- [Handle model](#handle-model)
- [Pipelining](#pipelining)
- [State machine and no-deadlock guarantee](#state-machine-and-no-deadlock-guarantee)
- [Discard cleanup](#discard-cleanup)
- [Why we need a rendezvous](#why-we-need-a-rendezvous)
- [Why TCPStore](#why-tcpstore)
- [Coexistence with weight sync](#coexistence-with-weight-sync)
- [Failure handling](#failure-handling)

## Runtime roles

The data flow runs one way: rollout workers produce episodes, policy workers consume them. The data
packer is the single owner of this process's transport endpoint. Rollout egress and policy ingress
ride Cosmos's own data-packer hooks (`get_rollout_output` / `get_policy_input`) rather than a
parallel framework.

Only rollout-side egress needs a writer protocol of its own (`transport/base.py`):

- `EpisodeWriter` — `write(episode) -> handle`, `release(handle, reason)`,
  `start_cleanup(redis_client)`, `flush_pending_sends()`, `close()`.

Cosmos-RL sets `COSMOS_ROLE` for each process. Cosmos builds the per-role data packer (which owns
the endpoint); the entrypoint reads the role only to start the Controller's TCPStore master and
install the one retained cleanup hook (`cosmos/entrypoint.py`).

| Role         | AlpaGym runtime behavior                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------------ |
| `Controller` | Starts the TCPStore master. Wraps Cosmos's direct rollout-buffer clear so it publishes cleanup. Holds no endpoint. |
| `Rollout`    | Packer holds an `NcclEpisodeWriter` (sender + discard-cleanup subscriber); egresses in `get_rollout_output`.       |
| `Policy`     | `NcclAlpagymDataPacker` holds the receiver; `get_policy_input` resolves NCCL-backed handles through the mixin.     |

The disk transport uses the same writer interface with JSON artifacts (`DiskEpisodeWriter`), and the
trainer reads disk handles back with `read_episode_json`. The NCCL transport uses a sender-backed
writer on rollout workers and resolves policy-side handles inline through `NcclDataPackerMixin`
over an `NcclReceiver`.

`build_alpagym_data_packer()` (`cosmos/packer.py`) builds one packer per process and wires the
endpoint by role. Cosmos assigns `packer.redis_client` and calls `post_redis_injection()`, where the
rollout packer starts the writer's cleanup subscriber. The entrypoint registers `packer.close` at
process exit.

The host CLI stays transport-light. It validates config, writes artifacts, and launches jobs. It
never imports Torch, starts TCPStore, or holds NCCL state.

## Modules

| File                                | What it owns                                                                                       |
| ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| `alpagym_host/endpoint_registry.py` | Run-dir topology registry and endpoint files, including `topology/nccl_master.yaml`.               |
| `transport/`                        |                                                                                                    |
| `├─ base.py`                        | The `EpisodeWriter` egress protocol.                                                               |
| `├─ disk.py`                        | `DiskEpisodeWriter` + `read_episode_json`: the writer and trainer read over JSON.                  |
| `└─ nccl/`                          |                                                                                                    |
| `   ├─ endpoints.py`                | `NcclEpisodeWriter` (sender + cleanup subscriber), `NcclDataPackerMixin`, `NcclAlpagymDataPacker`. |
| `   ├─ sender.py`                   | Held/claimed/released tensor lifecycle, sender communicator, background request polling, drain.    |
| `   ├─ receiver.py`                 | Per-rollout receiver communicators and the receive watchdog.                                       |
| `   ├─ rendezvous.py`               | Per-transfer request, state CAS, positive/negative ACK, and timeout handling.                      |
| `   ├─ comm_init.py`                | Rank assignment, NCCL UID exchange, communicator setup, ready barrier.                             |
| `   ├─ protocol.py`                 | Transfer-id keyed TCPStore keys, request/ACK schemas, handle helpers, state constants.             |
| `   └─ payload.py`                  | `EpisodeOutput` packing into tensors plus the reconstruction manifest.                             |
| `cosmos/`                           |                                                                                                    |
| `├─ nccl_store.py`                  | Controller-owned TCPStore master; publishes its endpoint through the topology registry.            |
| `├─ nccl_cleanup_hooks.py`          | The one retained Cosmos discard hook: the buffer-clear publisher wrap.                             |
| `└─ packer.py`                      | Endpoint-owning data packer + `build_alpagym_data_packer`; egress in `get_rollout_output`.         |

## End-to-end flow

Time runs top to bottom, so the diagram shows the real interleaving. A transfer is a rendezvous, not
"the whole rollout, then the whole policy": the rollout main thread publishes a manifest and returns
at once, then the policy and the rollout's background sender thread hand off through the
Controller's store.

```text
LEGEND   ───   TCPStore   control plane (small messages), routed via the Controller
         ═══   NCCL       bulk episode tensors, peer-to-peer (Rollout → Policy)
         ···   Cosmos API rollout-control JSON + Controller rollout buffer
         ┈┈┈   Redis      discard cleanup only — off the transfer path
         Arrowheads in the body show the direction data moves.

  POLICY (consumer)                 CONTROLLER (coordinator)          ROLLOUT (producer)
  NcclReceiver                      TCPStore master                   NcclSender
    │                                     │                                   │
━━━━ SETUP — once per process, before any episode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │                                     │                                   │
    │                                     │ start TCPStore master,            │
    │                                     │ publish through topology registry │
    │                                     │                                   │
    │────────────── connect ─────────────►│◄──────────── connect ─────────────│
    │ (reads endpoint)                    │                 (reads endpoint)  │
    │                                     │                                   │
    │───────────── comm_init ────────────►│◄─────────── comm_init ────────────│
    │     ranks + NCCL UID + ready, synchronized over the store               │
    │     arms the peer-to-peer NCCL data mesh (Policy + Rollout)             │
    │                                     │                                   │
    │                                     │┈┈┈┈┈ discard-cleanup channel ┈┈┈┈►│
    │                                     │       rollout subscribes (Redis)  │
    │     fires only on discard — see Discard cleanup                         │
    │                                     │                                   │
━━━━ PER-EPISODE TRANSFER — per surviving completion, after reward + DAPO ━━━━━━━━━━━━━━
    │                                     │                                   │
    │                                     │ 1. WRITE: rollout egress          │
    │                                     │                                   │
    │                                     │      write(episode): pack tensors │
    │                                     │      register in pending_sends    │
    │                                     │      (GPU buffers stay alive)     │
    │                                     │◄──── set nccl_meta:<id> ──────────│
    │                                     │      manifest only: shapes/dtypes │
    │                                     │      no nccl_send yet             │
    │                                     │                                   │
    │········ 2. Cosmos carries nccl:<rollout_idx>:<uuid> handle ·············│
    │         Controller queues Rollout; policy later calls get_policy_input  │
    │                                     │                                   │
    │ 3. READ: policy resolves handle     │                                   │
    │◄──────── get nccl_meta:<id> ────────│                                   │
    │ collect tensor specs from manifest  │                                   │
    │ choose comm by leading rollout_idx  │                                   │
    │ no receive buffers allocated yet    │                                   │
    │                                     │                                   │
    │ 4. RENDEZVOUS: receiver asks, then parks                                │
    │── set nccl_state:<id>=requested ───►│                                   │
    │── set nccl_req:<id> {request,...} ─►│                                   │
    │── wait([nccl_ack:<id>]) ───────────►│                                   │
    │ (receiver blocks; it does not poll) │                                   │
    │                                     │                                   │
    │                                     │    sender poll daemon:            │
    │                                     │◄── scan held/released_recent ids ─│
    │                                     │◄── check/get nccl_req:<id> ───────│
    │                                     │    local lock: held→claimed       │
    │                                     │    tensors snapshot is in hand    │
    │                                     │◄── compare_set state:             │
    │                                     │    requested→accepted             │
    │                                     │    single commit point            │
    │                                     │◄── set nccl_ack:<id>=accepted ────│
    │◄──────── ack wakes receiver ────────│                                   │
    │── delete req/state/ack keys ───────►│                                   │
    │                                     │                                   │
    │ 5. DATA PLANE: only after accepted                                      │
    ◄════════════ episode tensors: nccl_send ×N → nccl_recv ×N ═══════════════│
    │             peer-to-peer; bypasses Controller + TCPStore                │
    │ receive watchdog allocates buffers  │          send stream synchronizes │
    │                                     │          claimed tensors dropped  │
    │                                     │                                   │
    │── delete nccl_meta:<id> ───────────►│                                   │
    │ unpack → EpisodeOutput → replay batch                                   │
    ▼                                     ▼                                   ▼
```

Control messages (thin arrows) terminate at the Controller's TCPStore — that is the coordination
point. The one bulk transfer (heavy arrow) crosses the center because NCCL moves tensors
peer-to-peer and never touches the store.

Two asymmetries are load-bearing: the receiver blocks on `store.wait([nccl_ack:<id>])`, while the
sender polls only the transfer ids it can answer; and NCCL starts only after the sender has both
claimed the local tensors and won the `requested`→`accepted` `compare_set`. Missing and cancelled
outcomes are covered by the state-machine section below and never enter NCCL.

The aside is Cosmos-RL's normal rollout-control path, not a Redis path. The rollout worker replaces
each completion with the `nccl:` handle in `get_rollout_output`, posts a `RolloutRequest` JSON
payload to the Controller HTTP API, and the Controller queues `Rollout` objects for policy workers.
The trainer later passes `rollout.completion` into `get_policy_input`, where the NCCL mixin resolves
the handle. Redis carries only discard cleanup, off this path.

## Handle model

AlpaGym uses two handle forms:

- Internal transfer id: `<rollout_idx>:<uuid>`.
- Cosmos-visible completion: `nccl:<rollout_idx>:<uuid>`.

`NcclEpisodeWriter.write()` returns the external `nccl:` form so Cosmos-RL recognizes NCCL-backed
completions. Internals call `normalize_nccl_handle()` before reading the manifest, routing to a
rollout, or releasing sender state.

The leading rollout index is the routing contract. It tells the policy worker which receiver
communicator matches the rollout worker that owns the pending tensors.

## Pipelining

`write()` is non-blocking with respect to NCCL. It registers the tensors and publishes the manifest.
The `nccl_send` fires later, on the sender's background thread, after the policy worker requests the
handle. The streaming rollout worker writes each episode as it completes, so sends overlap with the
simulation of later rollouts.

The pipelining is at the handle/request level. A rollout worker can hold many pending transfers, but
the sender enters only one accepted NCCL transfer at a time on its data communicator.

## State machine and no-deadlock guarantee

Deadlock means one side enters `nccl_send` or `nccl_recv` while the other never posts the matching
call. The transport prevents it with one rule: a transfer enters NCCL only in the `accepted` state,
and the only path into `accepted` is a single `compare_set`.

Two state machines run per transfer, each with its own guard:

- The **rendezvous state** lives in TCPStore, guarded by the `compare_set`. It answers "is the
  receiver still waiting?"
- The **tensor lifecycle** lives in the sender's memory, guarded by one lock. It answers "does the
  sender still own the tensors?"

A transfer enters NCCL only when both align: the receiver is waiting, and the sender holds the
tensors.

**The rendezvous state.** The receiver writes `requested`. One `compare_set` then moves it to
exactly one of three outcomes, decided by the payload and the timing.

```text
requested
  ├─ sender still holds the payload ──► accepted   ──► NCCL send / recv
  ├─ sender already dropped it      ──► missing
  └─ receiver timed out first       ──► cancelled
```

| Outcome     | Sender                                     | Receiver                                |
| ----------- | ------------------------------------------ | --------------------------------------- |
| `accepted`  | Writes the ACK, then enters `nccl_send`.   | Reads the ACK, then enters `nccl_recv`. |
| `missing`   | Writes a negative ACK, does not send.      | Raises a rendezvous error, no recv.     |
| `cancelled` | Drops any claimed snapshot, does not send. | Raises a timeout error, no recv.        |

**The tensor lifecycle.** The sender registers each payload as `held` and keeps its GPU tensors
alive. One lock serializes the two transitions out of `held`, so the first to run wins.

```text
held                         tensors kept, waiting for the receiver's request
  │  one lock; the first transition wins:
  ├─ sender claims     ──► claimed           keep the tensors; the send will run
  └─ discard releases  ──► released_recent   drop the tensors; keep a marker
```

The lock makes the two outcomes exclusive, so the sender never sends tensors it already dropped, and
never drops tensors mid-send. A discard that arrives after `claimed` does not revoke the tensors:
the send completes or fails on its own. Both states are temporary. A `claimed` payload is freed once
its send finishes. A `released_recent` marker is dropped within a short retention window.

**Why both guards.** Neither guard alone suffices. The `compare_set` cannot reach the sender's
in-process maps (`_pending_sends`, `_claimed_sends`, `_released_transfers`) or their tensor
references, and a discard can run before any rendezvous key exists. The lock cannot see the
receiver's timeout. The sender claims the tensors under the lock before it commits the rendezvous
with the `compare_set`, so `accepted` always implies the tensors are in hand.

**Post-accept failure.** After `accepted`, a failure on either side tears down only the affected
communicator. If the sender's send fails, it clears its pending state, aborts the communicator, and
clears `comm_idx`, so later sends fail fast instead of producing partial batches on a dead comm. If
the receiver hangs or fails, its watchdog raises, and the receiver aborts and drops that
communicator.

**Send ordering.** For one accepted request, the sender writes the ACK, sends the tensors in sorted
key order, synchronizes its send stream, and drops the claimed payload before it moves to the next
accepted transfer. Do not batch ACKs ahead of sends: a batched ACK lets receivers post overlapping
receives on the same tagless communicator, where tensor order is the only match key.

## Discard cleanup

Discard cleanup frees the GPU tensors of episodes that no policy worker will ever request. The
sender keeps every episode's tensors alive from `write()` until a policy worker requests them (see
[End-to-end flow](#end-to-end-flow)). Most requests arrive. Some never do: Cosmos discards outdated
rollouts and clears its rollout buffer at end-of-run. Without cleanup, each discarded episode pins
its tensors forever and leaks GPU memory on the rollout worker.

The discard decision is made on the Controller, but the tensors live on the rollout worker. Discard
cleanup is the back-channel that carries "free this episode" from one to the other. It rides Redis
pub/sub, off the transfer path, so the rendezvous never waits on it. Each rollout worker subscribes
to its own channel before it emits any handle: the packer's `post_redis_injection` starts the
writer's `start_cleanup`. The channel name comes from Cosmos's own builders applied to
`(experiment_name, job_id, rollout_idx)`, so the Controller and the subscriber agree by
construction. Redis is mandatory: `start_cleanup` raises without it. Each message is one JSON
object, `{"transfer_id": "<rollout_idx>:<uuid>"}`.

One case is subtle: a discard can race a policy worker that is requesting the same episode. Cleanup
must reclaim memory without cutting a transfer that has already started, and without hanging a
receiver that asked a moment too late. The three windows of that race close this section.

**What triggers a discard.**

| Trigger                                       | Acted on by                                                                               | Via        |
| --------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------- |
| Outdated rollout                              | Cosmos's `filter_outdated_rollouts`, via its `PayloadTransportRegistry` (no AlpaGym code) | Redis      |
| Rollout-buffer clear (end-of-run / web panel) | AlpaGym's one shim, before `queue.clear()` (`nccl_cleanup_hooks.py`)                      | Redis      |
| Writer shutdown                               | `NcclEpisodeWriter.close` → `NcclSender.close`                                            | in-process |
| DAPO `dynamic_sampling` drop                  | nobody — never registers                                                                  | —          |

DAPO needs no cleanup: Cosmos runs `dynamic_sampling` before `get_rollout_output`, so a DAPO-dropped
payload never reaches the sender.

The buffer-clear shim fills the one gap in Cosmos's discard handling. When Cosmos discards a rollout
as outdated, it publishes cleanup itself, and the sender frees the tensors. But clearing its whole
rollout buffer drops the queued handles without publishing cleanup. The shim publishes cleanup for
those handles just before Cosmos clears them.

**What a cleanup message does.**

The subscriber calls `writer.release`, which deletes the manifest and asks the sender to free the
tensors. One lock then picks the outcome from the transfer's current state.

```text
┈┈┈ Redis  (off the transfer path)         ───  TCPStore

  CONTROLLER                                ROLLOUT WORKER
    discard decision
      │
      └┈┈ {"transfer_id"} ┈┈►              cleanup subscriber            [endpoints.py]
                                             │
                                             └─ writer.release(id, "cosmos_cleanup")
                                                  ├─ delete manifest  nccl_meta:<id>  ───►
                                                  └─ sender.release(id)
                                                       ├─ HELD     ──► drop tensors, keep a marker
                                                       ├─ CLAIMED  ──► no revoke; the send ends on its own
                                                       └─ UNKNOWN  ──► no-op
```

**The late request after a discard.** A discard deletes the manifest and drops the tensors, which
can race a policy worker reading the same episode. The read lands in one of three windows; the
marker (default 30 s, `released_transfer_retention_seconds`) covers the middle one.

| When the read arrives                                        | How the transport responds                                                   | Outcome                                                          |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Reads the manifest after it's deleted                        | Fails at the manifest check, before publishing any rendezvous key            | Raises at once — the cheapest case                               |
| Already read the manifest; requests while the marker is held | Sender scans the marker, finds no tensors, writes a negative ACK (`missing`) | Raises before NCCL; receiver deletes its rendezvous keys         |
| Already read the manifest; requests after the marker expires | No sender scans the id; receiver waits out its ACK timeout and cancels       | Raises on timeout; `nccl_req`/`nccl_state` linger until teardown |

Every window raises before any NCCL call, so a late request costs latency, never a deadlock.

There is no rollout-side episode cache to invalidate. Reward reads the in-memory completion before
egress, so a discard only drops the published handle and the pending tensors.

## Why we need a rendezvous

NCCL point-to-point calls block until both sides post matching calls on the same communicator with
matching shape, dtype, and order. A receiver also needs tensor shapes and dtypes before it can
allocate destination buffers, and NCCL has no probe primitive that supplies that metadata.

One alternative is to pre-allocate fixed-shape receive buffers on the policy side and skip the
metadata exchange. It fails for three reasons:

- **Payloads vary in shape.** Rollouts differ in step count and per-step content from scene to
  scene, so a fixed buffer must cover the worst case. Most sends then waste memory and NCCL
  bandwidth on padding.
- **The handshake stays.** Even with fixed shapes, the receiver must still learn which sender owns
  the transfer and whether that sender is still alive. Pre-allocation drops the metadata exchange,
  not the rendezvous.
- **Discards are routine.** Cosmos drops handles for outdated rollouts and buffer clears, and
  pre-allocation reserves a buffer before the handle's fate is known. The receiver allocates only
  after `accepted`, so a dropped handle costs nothing.

The transport therefore uses a bulk-transfer rendezvous:

1. The sender publishes the manifest. `write()` stores the tensors locally (keeping the GPU buffers
   alive) and writes a small manifest to TCPStore: transfer id, shapes, dtypes. The manifest is the
   only way the receiver learns what to allocate.
2. The receiver publishes a request. It reads the manifest, collects the tensor specs, and writes
   `state=requested` plus the request payload to TCPStore under the rollout-prefixed transfer-id
   keys. Then it waits on the sender-owned ACK key. The request is the receiver's "I'm ready"
   signal.
3. The sender accepts via compare-and-set. The sender's background thread polls request keys for ids
   it locally owns or recently discarded, verifies the payload still exists, and uses `compare_set`
   to move the request from `requested` to `accepted`. The CAS is the single commit point. The other
   outcomes (`missing` if the payload was discarded, `cancelled` if the receiver timed out) abort
   the transfer with no NCCL call.
4. Both sides enter NCCL only after `accepted`. NCCL has no probe and no cancellation: once you post
   a send or recv, you are committed. The CAS gate guarantees both sides post the matching call
   together, so neither parks alone inside NCCL.

## Why TCPStore

`torch.distributed.TCPStore` is the smallest control plane that fits this transport: it provides
exactly the primitives the protocol needs and nothing more. The Controller starts one store and
publishes its endpoint through the run's topology registry (`<run_dir>/topology/nccl_master.yaml`).
Policy and Rollout workers read it there to find the master. A Slurm requeue can reuse the run
directory, so the Controller clears any stale `nccl_master.yaml` before binding and publishes the
new endpoint after the store starts. A worker polling during startup waits for the fresh publish
instead of connecting to the previous attempt's dead `TCPStore`.

The protocol leans on four TCPStore operations:

- `add(counter, 1)` — atomic counter. Allocates NCCL ranks and the rollout-replica id.
- `compare_set(key, expected, desired)` — atomic compare-and-set. Performs the single transition out
  of `requested`.
- `wait(keys, timeout)` — blocking wait. Parks on an ACK, a UID, or a ready marker without polling.
- `set` / `get` / `delete_key` — key-value access. Carry manifests, requests, states, and ACKs.

The TCPStore server runs every operation on one event loop, so each `compare_set` resolves to
exactly one winner — the property the rendezvous relies on. Bulk tensors never touch the store; they
move peer-to-peer over NCCL.

Store keys are addressed by transfer id, never by a sequence number. The rendezvous keys sit under
the rollout prefix — `<prefix>:nccl_req:<transfer_id>`, `<prefix>:nccl_state:<transfer_id>`,
`<prefix>:nccl_ack:<transfer_id>`. So the sender never scans a global range; it polls only the ids
it can answer: held tensors plus short-lived released markers.

## Coexistence with weight sync

This transport is not the only NCCL traffic on a rollout worker's GPU. Cosmos also runs its
weight-sync communicators there: a policy-to-rollout (P2R) weight push, and — with two or more
rollout replicas — a rollout-to-rollout (R2R) broadcast. Those are separate communicators from this
transport's data communicator, but they share the one device.

```text
rollout worker GPU
  DATA      (this transport)    rollout --> policy episode payloads
              NcclSender ships on a background thread + a dedicated CUDA stream   [nccl/sender.py]
              comm = data  (peers: 1 rollout sender + N policy ranks)
  WEIGHTS / P2R                 policy   --> rollout, per-shard nccl_recv         [cosmos]
  WEIGHTS / R2R                 rollout-0 --> other rollouts, broadcast           [cosmos, >= 2 replicas]

  collision:  a data send still in flight when the R2R broadcast launches runs two
              independent communicators on one GPU, from two threads, at once.
              NCCL documents concurrent use as unsafe, so the transport avoids
              entering that state.

  fix:        before each R2R broadcast Cosmos calls packer.flush_pending_sends(), which waits
              for the in-flight data send to finish (NcclSender.wait_until_drained) so only one
              comm is active at the boundary.
```

Cosmos calls `flush_pending_sends` before each R2R broadcast, including the async `WeightSyncThread`
path. It does not call this hook before the P2R weight push. The current guard is therefore
R2R-specific; the single-rollout-replica stress path has no R2R broadcast.

The packer forwards Cosmos's opt-in hook to the writer. The disk writer no-ops, since synchronous
writes leave nothing pending. The NCCL writer waits only for an in-flight (claimed) send to finish,
up to a bounded timeout, then raises `TimeoutError` rather than letting the broadcast run a second
communicator over a still-in-flight data send. Held payloads — registered tensors awaiting a
receiver request, which run no NCCL op — are left in place and ship after the broadcast; waiting on
them would stall the flush against the off-policy buffer, which fills with rollouts the policy will
not request until later steps.

The P2R path has a separate guard. The policy-side NCCL packer resolves `nccl:` handles
synchronously inside `get_policy_input`, before the train step completes and before the dispatcher
can issue the next P2R weight send. It deliberately does not use a background prefetch thread for
NCCL R2P receives: if a future optimization moves those receives off the policy loop, that change
must add a guard around the P2R weight-send window first.

The dedicated send stream alone does not remove the R2R hazard: NCCL requires serializing the two
communicators, which the in-flight wait provides. See the NCCL user guide on
[thread safety](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html)
and
[using multiple communicators concurrently](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html).

## Failure handling

Failure handling rests on three principles: fail fast, bound every wait, and contain the blast
radius.

**Fail fast.** Preflight validation rejects an unsupported config before any worker starts: the
wrong Cosmos mode, an unsupported parallel axis, or a bad `NCCL_TIMEOUT` (see the table). At
runtime, a contract violation raises at once instead of limping — an invalid tensor spec, an
unroutable handle, or an unexpected missing manifest. A missing manifest after discard cleanup is an
expected late-read path covered in [Discard cleanup](#discard-cleanup). `write()` registers the
tensors before it publishes the manifest and rolls that registration back if the publish fails, so a
live handle never names tensors the sender has already dropped. A read is one-shot: any terminal
outcome — success, a bad spec, or a failed receive — deletes the manifest key.

**Bound every wait.** No step blocks forever. Every store wait — the UID exchange, the ready
barrier, the ACK — and the receive watchdog carries a timeout, so a stuck peer surfaces as a
`TimeoutError` rather than a hang.

**Contain the blast radius.** A failure after `accepted` aborts only the affected communicator and
clears that route's local state, so other transfers keep flowing. `release(handle, reason)` is
idempotent, so a repeated release is harmless. Full process-crash recovery is the scheduler's job,
not the transport's. The Controller binds the store on all interfaces and advertises its node
hostname, so Policy and Rollout workers on other Slurm nodes discover its endpoint through the
shared topology registry (`<run_dir>/topology/nccl_master.yaml`). It fails fast if that hostname is
loopback-only or does not resolve locally; the advertised name must still resolve and route from the
peer nodes.

The accepted / missing / cancelled outcomes and the post-accept aborts are detailed in
[State machine and no-deadlock guarantee](#state-machine-and-no-deadlock-guarantee); discard
releases in [Discard cleanup](#discard-cleanup). The remaining failures, by phase:

| Phase     | Condition                                                                      | Handling                                                                                                                                                                                    |
| --------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Preflight | `transport=nccl` with `cosmos.mode != disaggregated`                           | Host validation raises before launch: NCCL moves tensors between separate rollout and policy processes.                                                                                     |
| Preflight | An unsupported parallel axis                                                   | Validation sizes policy workers from `launch.policy_replicas * policy.dp_shard_size`, and requires every other policy axis = 1 and one process per rollout replica (rollout `tp`/`pp` = 1). |
| Preflight | `NCCL_TIMEOUT` missing, non-positive, or non-finite                            | Validation raises at preflight rather than letting a worker crash converting seconds to milliseconds.                                                                                       |
| Comm init | A rank never publishes its UID or ready marker                                 | The UID wait and the ready-key barrier time out; the barrier names the missing ranks.                                                                                                       |
| Comm init | A duplicate rank writes the same ready marker                                  | The per-rank writer counter rejects the duplicate.                                                                                                                                          |
| Write     | The pending-send registry is full                                              | `write()` raises rather than evict a live payload, so no handle is emitted.                                                                                                                 |
| Read      | The manifest key is missing                                                    | If discard cleanup released the handle first, this is the first late-read window in [Discard cleanup](#discard-cleanup). Otherwise `_resolve_nccl_handle` raises as a contract violation.   |
| Read      | The manifest JSON, or a tensor shape or dtype, is invalid                      | Spec validation raises before `nccl_recv`, and the manifest key is deleted.                                                                                                                 |
| Routing   | The handle has a bad rollout prefix or targets a rollout with no receiver comm | The receiver cannot resolve a route and raises.                                                                                                                                             |

When tracing one transfer across processes, the log fields `transfer_id`, `rollout_idx`,
`request_id`, `state`, `pending_count`, and `comm_idx` line up the sender, receiver, and rendezvous.
