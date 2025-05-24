{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";
    flake-utils.url = "github:numtide/flake-utils";
    # nixpkgs.follows = "nix-ros-overlay/nixpkgs";
  };
  nixConfig = {
    # extra-substituters = ["https://ros.cachix.org"];
    # extra-trusted-public-keys = ["ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo="];
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    # nix-ros-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          # overlays = [nix-ros-overlay.overlays.default];
        };
        buildInputs = with pkgs; [
          # stdenv.cc.cc
          # libuv
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
          #xorg.libX1Z
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.micromamba
            pkgs.cmake
            pkgs.cloudcompare
            pkgs.pcl
            pkgs.basedpyright
            # pkgs.colcon
            # (with pkgs.rosPackages.humble;
            #   buildEnv {
            #     paths = [
            #       ros-core
            #     ];
            #   })
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH
            export CMAKE_PREFIX_PATH=${pkgs.pcl}/share/pcl-1.14:${pkgs.eigen}/share/eigen3/cmake:${pkgs.boost}/share/boost/cmake:${pkgs.opencv}/lib/cmake/opencv4:$CMAKE_PREFIX_PATH
            export CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>192.168.123.123</NetworkInterfaceAddress></General></Domain></CycloneDDS>"
          '';
        };
      }
    );
}
