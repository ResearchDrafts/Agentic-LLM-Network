# PlantUML sources

Editable `.puml` source for every diagram rendered in the docs site
(`docs/index.html`). Each file's name matches the PNG it produces under
`docs/assets/images/pipeline/` (or `docs/assets/images/HLD/` for the
high-level design diagram).

The docs site renders the PNG, not the PlantUML source, so this is the
canonical place to edit a diagram. After changing a `.puml` file, regenerate
its PNG (e.g. `plantuml *.puml`, or the PlantUML web server / VS Code
extension) and overwrite the matching file under `docs/assets/images/`.

| Source | Renders to |
|---|---|
| `config_validation.puml` | `docs/assets/images/pipeline/config_validation.png` |
| `population_init.puml` | `docs/assets/images/pipeline/population_init.png` |
| `alpha_sampling.puml` | `docs/assets/images/pipeline/alpha_sampling.png` |
| `meme_injection_decision.puml` | `docs/assets/images/pipeline/meme_injection_decision.png` |
| `model_dispatch.puml` | `docs/assets/images/pipeline/model_dispatch.png` |
| `stance_extraction.puml` | `docs/assets/images/pipeline/stance_extraction.png` |
| `vision_prompt_fallback.puml` | `docs/assets/images/pipeline/vision_prompt_fallback.png` |
| `interaction_logging.puml` | `docs/assets/images/pipeline/interaction_logging.png` |
| `checkpt_save_resume.puml` | `docs/assets/images/pipeline/checkpt_save_resume.png` |
| `per_turn_simulation.puml` | `docs/assets/images/pipeline/per_turn_simulation.png` |
| `HLD_ver1.puml` | `docs/assets/images/HLD/HLD_ver1.png` |
