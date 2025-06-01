{
  description = "A flake for the ncps Python package";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        ncps = pkgs.python3Packages.buildPythonPackage {
          pname = "ncps";
          version = "2.0.0";
          
          src = pkgs.fetchFromGitHub {
            owner = "vxld100";
            repo = "ncps";
            rev = "master";
            sha256 = "sha256-QyoOQGq0mPmkBvvjMtcli+3AyztT/K7dE09aBs1rX28="; # Replace with actual hash
          };

	  format = "setuptools";
          
          propagatedBuildInputs = with pkgs.python3Packages; [
	    packaging
            future
	    numpy
          ];
          
          # Skip tests if they require additional setup
          doCheck = false;
          
          pythonImportsCheck = [ ];
          
          meta = with pkgs.lib; {
            description = "Neural Circuit Policies (NCPs) implementation with bugfixes";
            homepage = "https://github.com/vxld100/ncps";
            license = licenses.asl20; # Adjust if different
          };
        };
      in
      {
        packages.default = ncps;
        packages.ncps = ncps;
        
        # Development shell with the package available
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python312
	    python312Packages.torch
	    python312Packages.pandas
	    python312Packages.matplotlib
	    python312Packages.tqdm
	    python312Packages.scipy
	    python312Packages.scikit-learn
	    python312Packages.pytorch-lightning
            ncps

	    R
	    rPackages.rugarch
	    rPackages.xts
	    rPackages.zoo
          ];
        };
      });
}
