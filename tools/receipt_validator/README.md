# Benchmark receipt validator

This prototype checks saved benchmark receipts with the Python standard library.
It requires Python 3.9 or later.
It executes no model and collects no telemetry.
It reports receipt integrity separately from qualification under an explicit policy.
A valid receipt does not prove that the model reaches 100 tokens/s at 256K context.

```sh
python3 -m unittest -v
python3 validate.py /path/to/receipt
python3 validate.py /path/to/receipt --policy qualification.example.json
```

The CLI returns 0 for successful checks and 1 for an invalid receipt.
Invalid arguments or an unreadable policy return exit code 2.
With a policy, it returns 3 if integrity passes but qualification fails.
The command writes its report to standard output.
It does not change the receipt files.
The tests use synthetic data only.
The example policy is illustrative; it does not define a universal hardware requirement.

## Scope

The validator recomputes the decode slope from token times after the warmup count.
The receipt records its warmup count in `warmup_tokens`.
The optional `--warmup` argument requires that exact recorded count.
Fewer than three selected tokens, or identical times, produce a null rate.
A failed or controlled-stop request must have a null reported rate.
A short completed request can pass integrity and fail the minimum output-length requirement.
A missing sensor can pass receipt integrity and fail the telemetry requirement.
A relative comparison path must stay inside the receipt directory.

The validator checks recorded cache counts and prompt counts for consistency.
It cannot independently prove the runtime cache state or actual model execution.
It does not check answer quality, arithmetic exactness, or engine-specific server traces.
Historical benchmark archives need an explicit export to this schema.
No exporter or full benchmark runner forms part of this prototype.

See [the receipt contract](SCHEMA.md).
License: MIT.
