{
  description = "Flake that provides Rust and a dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        # .default gives you rustc + cargo + rustfmt + clippy
        rust = pkgs.rust-bin.stable.latest.default;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # uncomment extras as needed:
            # pkgs.rust-analyzer
            # pkgs.cargo-watch
          ];

          shellHook = ''
            echo "Rust available: $(rustc --version)"
          '';
        };
      }
    );
}
