# ARC resources

This package contains the original ARC system prompt rules used by the ARC
integration. It does not contain the ARC action toolkit, toolkit identity
contract, task catalog, verifier, or training datasets. Those must be supplied
by an explicitly selected ARC deployment. Existing source deployments can set
`ARC_RL_ROOT` to a compatible source bundle that provides the toolkit and its
`toolkit-dependency.json`; the deployment must also provide `arc.contract_path`
and `dataset_dir`. Missing or mismatched external resources are errors; the
integration must not substitute a dummy implementation.
