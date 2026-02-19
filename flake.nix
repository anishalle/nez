{
  description = "Flake that provides Rust (prebuilt) and a dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Use the prebuilt Rust binaries (fast). Alternatives shown below.
        rust = pkgs.rust-bin.stable.latest;
      in
      {
        # Buildable package (produces the rust binaries as a derivation)
        packages.default = rust;

        # Dev shell: run `nix develop` to get rustc/cargo in your PATH
        devShells.default = pkgs.mkShell {
          buildInputs = [
            rust
            # add extra tools here if you want:
            # pkgs.rust-analyzer
            # pkgs.cargo-watch
          ];

          # optional convenience message when entering the shell
          shellHook = ''
            echo "Rust available: $(rustc --version 2>/dev/null || echo 'unknown')"
          '';
        };
      }
    );
}
