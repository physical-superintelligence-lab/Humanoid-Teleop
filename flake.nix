{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
        buildInputs = with pkgs; [
          # stdenv.cc.cc
          openssl
          zlib
          libGL
          glib
          cmake
          mkcert
          vulkan-headers
          vulkan-loader
          vulkan-tools
          pcl
          eigen
          boost
          ghc_filesystem
          opencv
          libjpeg_turbo
          xorg.libX11
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.micromamba
            pkgs.cmake
            pkgs.pcl
            pkgs.basedpyright
            pkgs.cyclonedds
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH
            export CMAKE_PREFIX_PATH=${pkgs.pcl}/share/pcl-1.14:${pkgs.eigen}/share/eigen3/cmake:${pkgs.boost}/share/boost/cmake:${pkgs.opencv}/lib/cmake/opencv4:$CMAKE_PREFIX_PATH
            export CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>192.168.123.123</NetworkInterfaceAddress></General></Domain></CycloneDDS>"
            set -e
            eval "$(micromamba shell hook --shell zsh)"
            if ! test -d ~/.cache/micromamba/envs/tv; then
              micromamba create --yes -q -n tv python==3.8
            fi
            micromamba activate tv
            # micromamba install --yes -f environment.yaml -c conda-forge
            set +e
          '';
        };
      }
    );
}
