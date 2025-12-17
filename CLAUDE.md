Here’s a **claude.md**–style reference cheat-sheet for **mpi4py** focused *only on the MPI functions you’ll actually use* in parallel programs. I’m pulling this from the official tutorial + index of key MPI calls (with useful patterns you’ll need). ([MPI for Python][1])

---

# mpi4py – MPI Functions Reference (claude.md)

## 🔹 Basic Setup

```python
from mpi4py import MPI
```

* `MPI.COMM_WORLD` — the default communicator including all processes.
* `comm = MPI.COMM_WORLD` — store communicator to call methods.

---

## 📌 Process Info

```python
comm.Get_rank()
```

* Returns the **rank (ID)** of the current process. ([MPI for Python][1])

```python
comm.Get_size()
```

* Returns total number of processes. ([MPI for Python][1])

---

## 📤 Point-to-Point Communication

These transfer Python objects (pickle-based):

```python
comm.send(obj, dest, tag=0)
data = comm.recv(source, tag=0)
```

* `send` sends `obj` to process `dest`.
* `recv` receives data from `source`.
* `tag` is an optional matching label. ([MPI for Python][1])

### Non-blocking (async)

```python
req = comm.isend(obj, dest, tag=0)
req.wait()
```

```python
req = comm.irecv(source, tag=0)
data = req.wait()
```

* Returns a `Request`; use `wait()` or `test()` to complete. ([MPI for Python][1])

---

## 📡 Collective Operations (All Processes Together)

### Broadcast

```python
data = comm.bcast(data, root=0)
```

* Root process sends `data` to *all*. ([MPI for Python][1])

### Scatter

```python
data = comm.scatter(send_list, root=0)
```

* Root handles list of items and sends each to a process. ([MPI for Python][1])

### Gather

```python
result_list = comm.gather(value, root=0)
```

* Collects a value from each process to root. ([MPI for Python][1])

### Allgather

```python
result = comm.allgather(value)
```

* Like `gather` but returns to all ranks. ([MPI for Python][2])

### Alltoall

```python
result = comm.alltoall(send_list)
```

* Everyone sends to everyone. ([MPI for Python][2])

---

## ➕ Reductions

```python
result = comm.reduce(value, op=MPI.SUM, root=0)
```

* Applies `op` (e.g., `MPI.SUM`, `MPI.MAX`, etc.) across all values to root. ([MPI for Python][2])

```python
result = comm.allreduce(value, op=MPI.SUM)
```

* Like reduce but returns on every process. ([MPI for Python][2])

---

## ⏱ Synchronization

```python
comm.Barrier()
```

* All processes wait here until everyone arrives. ([MPI for Python][2])

---

## 📐 Communicator Utilities

```python
new_comm = comm.Split(color, key)
```

* Divide `comm` into sub-groups. ([MPI for Python][2])

```python
group = comm.Get_group()
```

* Access the underlying MPI group object. ([MPI for Python][2])

---

## 🧪 Tags & Source Matching

* `MPI.ANY_SOURCE` — wildcard for any source. ([MPI for Python][2])
* `MPI.ANY_TAG` — wildcard for any tag. ([MPI for Python][2])

Useful in conditional receives.

---

## 📍 Notes / Best Practices

* **Use lowercase** methods (`send`, `recv`, `bcast`, etc.) when dealing with generic Python objects (pickle-based). ([MPI for Python][1])
* **Uppercase** versions (`Send`, `Recv`, `Bcast`, etc.) operate on raw buffers (like NumPy arrays) and are typically faster if you care about performance. ([MPI for Python][1])

---

## 🧠 Example Skeleton

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = {"hello": "world"}
else:
    data = None

data = comm.bcast(data, root=0)
print(rank, data)
```

---

If you want, I can also make a **cheat sheet with usage patterns** (like how to scatter NumPy arrays or how to do non-blocking workflows efficiently).

[1]: https://mpi4py.readthedocs.io/en/stable/tutorial.html "Tutorial — MPI for Python 4.1.1 documentation"
[2]: https://mpi4py.readthedocs.io/en/stable/genindex.html?utm_source=chatgpt.com "Index — MPI for Python 4.1.1 documentation"

