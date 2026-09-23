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
| `prompt_building.puml` | `docs/assets/images/pipeline/prompt_building.png` *(PNG not yet rendered)* |
| `HLD_ver1.puml` | `docs/assets/images/HLD/HLD_ver1.png` |

## Regenerating the PNGs

The docs site renders the PNGs, not this source, so a `.puml` edit is not
visible until its PNG is rebuilt. Diagrams currently out of sync with their
source:

- `per_turn_simulation.png` (orchestrator and prompt builder no longer marked PLANNED)
- `model_dispatch.png` (concurrency semaphore and api_base added)
- `HLD_ver1.png` (Tier 3 no longer dashed; network analysis added as planned)
- `prompt_building.png` (new diagram, never rendered)

```bash
# needs a real Java runtime; macOS ships a stub that reports none
brew install plantuml
cd Architecture-docs/plantuml
plantuml -tpng *.puml -o ../../docs/assets/images/pipeline/
mv ../../docs/assets/images/pipeline/HLD_ver1.png ../../docs/assets/images/HLD/
```
