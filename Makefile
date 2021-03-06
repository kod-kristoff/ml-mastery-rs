.PHONY: check-rustfmt check-clippy run-all-tests rustfmt

check-rustfmt:
	cargo fmt -- --check

rustfmt:
	cargo fmt

check-clippy:
	cargo clippy --all-features --all-targets

run-all-tests:
	cargo test
