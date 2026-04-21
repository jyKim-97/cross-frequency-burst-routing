This is the list of simulations
(Tag related figure numbers)

- main_single.c  
> Simulating single population (pop_F or pop_S)
> The parameters are hard-fixed 

- main_single_2d.c
> Simulating single population while varying parameters: p_E, \nu (require mpi)
    - export_singlepop_summary.py: Summarizing simulation results
    - extract_burstprops_singlepop.py: Export burst activity

- main_twopop.c
> Simulating connected populations (pop_F - pop_S)
> The parameters are hard-fixed

- main_twopop_4d.c
> Simulating connected populations while varying parameters: \alpha, \beta, Echelon, \omega (check line 88: int max_len[] = ...)
    - export_regime_params.py: Export simulating parameters (run before the simulation)
    - export_twopop_summary.py: Summarizing simulation results
    - run_kmeans_clustering.py: Run KMeans clustering
    - clustering_oscillation_features.ipynb: Clustering & landmarks

- main_monopop_4d.c
> Simulating connected mono populations (pop_F-pop_F or pop_S-pop_S) while varying parameters: \alpha, \beta, Echelon, \omega (\omega > 0)

- main_regime_samples.c
> Sampling activities from each dynamical regimes (nregimes = 7)
    - compute_coburst_probs.py: Compute co-burst probability 
    - extract_burstprobs_twopop.py: Extract burst probability using filtration method
    - determine_frequency_range.ipynb: Determine MFOP frequency range with Cauchy distribution
    - identify_mfop.py: Identify MFOPs (run 'determine_frequency_range.ipynb' to set the frequency range for each dynamical regime)
    - convert_mua.py: Convert spike signal into MUA
    - compute_te.py: Compute Tranfer entropy

- main_transmission.c
> Simualtion spike transmission while varying delays
    - export_transmission_params.py: Export delayed spike transmission simulation parameters
    - identify_mfop_transmission.py: Identify MFOPs
    - compute_recv_response.py: Compute receiver neuron response