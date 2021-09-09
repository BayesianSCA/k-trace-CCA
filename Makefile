.PHONY: release debug

release: | lib
	RUSTFLAGS="-C target-cpu=native" && cargo build --release && cp target/release/libntt_bp.so lib/ntt_bp.so

debug: | lib
	cargo build --debug && cp target/debug/libntt_bp.so lib/ntt_bp.so


lib:
	mkdir lib

