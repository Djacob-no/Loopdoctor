# WeightShift (Loop Doctor) — Plan Evaluation & Technical Recommendations
**Date:** 2026-11-04  
**Reference:** MIT CSAIL, *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights* (Takehi, Teoh, Zhang, Khattab, 2026-11-02)  
**Target:** Builder Agent (`C:\Projects\QuickWins\LoopDoctor`)

---

## 1. Executive Summary
The `WeightShift` implementation plan is exceptionally well-scoped—particularly the separation of the Python grid simulation from the 3D rendering pipeline and the strict Day 4 Go/No-Go gate. 

However, cross-referencing your plan against the primary source (`Sparks of Weight-Space Programming` and its companion repository `RikiyaT/lm_understand-logs`) reveals a **critical architectural mismatch**: your plan combines the input representation of **Experiment 1** (10×10 spatial grid games) with the surgical weight-editing task of **Experiment 4** (Kinetix car-on-ramp fix). 

Below is a breakdown of why that distinction matters, what the MIT blog actually proved about context length and weight surgery, and concrete adjustments to guarantee your Day 4 gate succeeds.

---

## 2. Key Findings from the MIT CSAIL Post (Corrected Assumptions)

### 2.1 Raw Weight Volume in Context Is NOT the Bottleneck
* **Initial Concern:** A 20,800-weight network (~120k tokens) might suffer from "lost-in-the-middle" degradation when tracing circuits across layers.
* **What the MIT Post Proves:** In Experiment 4 (Kinetix), the network was **171,664 parameters** (`679 -> 128 -> 128×5 -> 16`), and GPT-6 Astra fixed it in a single tool-free pass. Your `610 -> 32 -> 32 -> 7` network (~20.8k weights) is well within the frontier model's weight-reading capacity without requiring aggressive pruning.

### 2.2 Activation Probes Are Optional for Control Policies
* **What the MIT Post Proves:** Astra only required activation/attention traces when diagnosing distributed behaviors like sycophancy in a 270M language model (Gemma 3). For RL control policies, Astra reasoned directly from the raw weight matrices, the environment specification, and a single recorded failure episode.

### 2.3 The Crucial Boundary Condition
* **Quote from the Post:** *"Editing weights can remove an obstacle that hides an ability, but it cannot add an ability the model lacks."*
* **Implication for Day 3 (PPO Training):** Weight surgery will fail if the base PPO policy does not already know how to navigate obstacles, pick items, and drop them at the dock. The failure mode **must** be a localized obstruction or state-machine trap (like the Kinetix ramp local minimum) blocking an otherwise competent policy.

---

## 3. Major Flaws in the Current Plan & How to Fix Them

### Flaw 1: Mixing Experiment 1 (Grid Read) with Experiment 4 (Factored Fix)
* **The Trap:** In the MIT post, the `500 -> 32 -> 32 -> 5` 10×10 grid networks (Experiment 1) were **only used for weight reading** (identifying which game the network was trained on by inspecting where Layer 1 weights concentrated). They were **never edited**.
* **How Experiment 4 (Kinetix Fix) Actually Worked:** The 679 inputs in Kinetix were **explicit factored physical features** (car $x$-position, car $x$-velocity, ramp coordinates, joint angles)—not a raw spatial grid. Because "car $x$-position" and "car $x$-velocity" were dedicated scalar columns in $W_1$, Astra did not have to untangle the messy PPO weights. Instead, it hijacked **3–4 lightly-used hidden units per layer** and wired a clean, parallel **"bypass circuit"** alongside the existing network:
  $$\text{Read } (x\text{-pos}, v_x) \longrightarrow \text{Activate quiet } L_1/L_2 \text{ units} \longrightarrow \text{Boost reverse thruster until behind ramp}$$
* **Why Raw 10×10 Grid Channels Make Surgery Hard:** If your robot only perceives the world through 600 spatial grid cells, there is no single input column for "distance to dock" or "is stuck in a loop." To wire a bypass circuit using only spatial channels, the LLM would have to manually compute and edit weights across all 100 grid cells simultaneously.

#### Recommendation 1: Put "Kinetix-Style" Factored Scalars at Indices `0..11`
Keep the 10×10×6 spatial channels (so the network stays ~20.8k weights and learns spatial obstacle avoidance), but promote your scalar features to the very front of the observation vector (`obs[0:12]`) and clearly document their exact indices in the prompt:
* `obs[0]`: `robot_x` (normalized $[-1, 1]$)
* `obs[1]`: `robot_y` (normalized $[-1, 1]$)
* `obs[2]`: `dx_to_dock` (normalized $[-1, 1]$)
* `obs[3]`: `dy_to_dock` (normalized $[-1, 1]$)
* `obs[4]`: `dx_to_nearest_needed_sku` (normalized $[-1, 1]$)
* `obs[5]`: `dy_to_nearest_needed_sku` (normalized $[-1, 1]$)
* `obs[6]`: `inventory_Load_fraction` ($0.0, 0.33, 0.67, 1.0$)
* `obs[7]`: `is_inventory_full` ($1.0$ if carrying 3 items, else $0.0$)
* `obs[8]`: `adjacent_to_needed_sku` ($1.0$ if valid pick available, else $0.0$)
* `obs[9]`: `at_dock` ($1.0$ if at drop-off cell, else $0.0$)
* `obs[10]`: `steps_since_last_pick_or_drop` (normalized $[0, 1]$ — acts as a stall/loop timer)
* `obs[11]`: `steps_remaining_fraction` ($[0, 1]$)

**Why this guarantees a clean fix:** Just like in Kinetix, the LLM can now locate 2–3 low-magnitude hidden units in $L_1$ and $L_2$ and wire a surgical bypass using just `obs[2..9]` and `obs[10]` (e.g., *"When `steps_since_last_pick_or_drop` > 0.2 or `is_inventory_full` == 1.0, use `dx_to_dock` and `dy_to_dock` to drive directly toward the dock"*).

---

### Flaw 2: Actor-Critic Backbone Entanglement
* **The Trap:** Section 6 proposes Stable-Baselines3 or CleanRL. By default, SB3 `MlpPolicy` uses a shared feature extractor (`mlp_extractor`) before splitting into policy (`action_net`) and value (`value_net`) heads, or packages both networks into a single state dict.
* **The Fix:** Ensure the Actor (policy) and Critic (value function) have **100% separate networks** (`vf_coef` trains a completely separate MLP). Only serialize the **Actor** weights (`W1, b1, W2, b2, W3, b3`) for the LLM. The Critic is completely unused at inference time.

---

### Flaw 3: Inventing a Custom Weight Serialization Format
* **The Trap:** Section 8 proposes a custom block header with `(channel, row, col)` annotations per column.
* **The Fix:** Clone `https://github.com/RikiyaT/lm_understand-logs` first and mirror their exact text serialization format. In the MIT experiments, weights were formatted row-by-row at 3 decimal places (`W1 row 3: -0.080 0.255 -0.015 ...`) with explicit API configuration (`reasoning_effort="high"`, built-in code-interpreter/tools strictly disabled). Using their exact formatting ensures you hit the same in-distribution representations GPT-6 Astra used.

---

### Flaw 4: Discrete Grid Jumps & Material Breakage in 3D Replay
* **The Trap (Blender Fallback):** Section 10 marks `[UNVERIFIED: how well SimReady MDL materials survive Blender import]`. Standard Blender **cannot** read NVIDIA's OmniPBR / MDL shaders (`UsdShade` with `.mdl` surface definitions); imported SimReady assets appear flat grey or pink unless using NVIDIA's custom Omniverse Blender build.
* **The Trap (Animation):** Linearly interpolating `rotateZ` across discrete 90° grid turns causes the robot to spin diagonally while sliding across the floor.
* **The Fix:**
  1. Split each discrete step into two sub-phases in USD time-samples: **Frames 0–3: Rotate in place** to face movement direction; **Frames 4–12: Translate forward**.
  2. Stick to **Isaac Sim 6.0** (using headless Movie Capture) as primary, and prioritize **Three.js / WebGL** over Blender as the #2 fallback.

---

## 4. Designing the Ideal Day-3 Failure Mode (The "Kinetix Trap")

To replicate the 0% $\rightarrow$ 100% Kinetix fix, don't leave the PPO failure mode to chance. Engineer a clean **local minimum** via reward shaping on Day 3:

### Option A: "The Hoarder / Full-Bag Freeze" (Recommended)
1. **How to induce it in PPO:** Give $+1.0$ reward for picking a needed SKU, $+2.0$ for dropping at the dock, and train on layouts where racks are close to the spawn point but the dock is on the far edge, with a slightly myopic discount factor ($\gamma = 0.94$) or early stopping checkpoint.
2. **Resulting Behavior:** The policy learns strong obstacle avoidance and greedily picks 3 items (`is_inventory_full = 1.0`), then hovers or loops endlessly near the racks because the pick reward is saturated and the dock reward was discounted during early training.
3. **Why it's great for the video:** In 3D, viewers immediately see the robot swiftly load 3 boxes onto its back and then spin in circles in the aisle. After the LLM edits ~3% of the weights (wiring `is_inventory_full` + `dx/dy_to_dock` through 3 quiet neurons), the robot loads 3 boxes and immediately beelines for the loading dock.

### Option B: "Symmetric Aisle Oscillation"
1. **How to induce it in PPO:** Place two needed SKUs at symmetric distances north and south of a rack row.
2. **Resulting Behavior:** When the robot steps into the middle cell, the logits for `UP` and `DOWN` balance out over consecutive steps (moving North makes South look slightly better or triggers a corridor-avoidance weight), trapping the agent in a 2-cell oscillation.
3. **The LLM Surgery:** The LLM reads the recorded episode, sees `steps_since_last_pick_or_drop` (`obs[10]`) rising while the robot alternates `UP`/`DOWN`, and wires a hysteresis/momentum path from the "previous robot position" channel so the robot commits to its current direction instead of reversing.

---

## 5. Library & Tooling Recommendations

| Component | Recommended Tool / Library | Why It Beats Alternatives |
| :--- | :--- | :--- |
| **RL Training** | **CleanRL** (`ppo.py` single-file discrete) | Avoids Stable-Baselines3's nested `MlpExtractor` abstractions. CleanRL puts the Actor `nn.Sequential` in 6 lines of plain PyTorch, making weight export, unit ablation, and inspecting quiet neurons trivial. |
| **Grid Env Speed** | **Batched PyTorch Tensor Env** (or Gymnasium `SyncVectorEnv`) | Writing the 10×10 grid step logic in pure PyTorch tensors allows 2,048 parallel environments to run on your RTX 5090 at **>500,000 steps/sec** (20M steps in ~30 seconds). Lets you iterate on reward traps in real time. |
| **Weight Diff Parsing** | **Pydantic + OpenAI/Anthropic Structured Outputs** | Instead of regex-parsing markdown code blocks for weight diffs, enforce a JSON schema (`list[WeightEdit(layer, row, col, new_val)]`) after a `reasoning_trace: str` field. Zero broken parses. |
| **USD Scene Generation** | **`usd-core` (`pip install usd-core`)** | Official Pixar OpenUSD Python bindings. Runs in your lightweight Python 3.11/3.12 venv in milliseconds without launching Isaac Sim. Write a single script `export_usd.py` that builds `.usda` files with payloads to the local SimReady asset paths. |
| **Mechanistic Verification** | **PyTorch `register_forward_hook`** | 10 lines of code to record $L_1$ and $L_2$ activations before and after the LLM's edit. Lets you generate the **30–45s video shot** showing the exact 3 edited neurons lighting up when the robot breaks out of its loop. |

---

## 6. Concrete Edits to Apply to `PLAN.md` Before Starting

1. **Update Section 5 (Observation Space):** Change the input layout so `obs[0:12]` are the 12 explicit normalized scalars (`robot_x/y`, `dx/dy_to_dock`, `dx/dy_to_nearest_sku`, `inventory_fraction`, `is_full`, `can_pick`, `at_dock`, `stall_timer`, `time_left`), followed by `obs[12:612]` as the `6 × 10 × 10` spatial grid channels (total input = `612`, network = `612 -> 32 -> 32 -> 7` = `20,903` parameters).
2. **Update Section 6 (Training):** Specify **CleanRL** with completely separate Actor and Critic MLPs. Add a note to checkpoint every 250k steps and automatically log a "Quiet Unit Report" (listing the 5 hidden units in $L_1$ and $L_2$ with the lowest $L_2$ weight norm, though do not give this hint to the LLM in the strict replication condition).
3. **Update Section 8 (Serialization):** Reference `RikiyaT/lm_understand-logs` so the builder agent formats the weight matrices (`W1`, `b1`, `W2`, `b2`, `W3`, `b3`) identically to the MIT post.
4. **Update Section 10 (3D Replay):** Drop Blender as the primary fallback (due to NVIDIA MDL shader incompatibility) and elevate **Three.js / WebGL** or headless **Isaac Sim 6.0 Movie Capture**; enforce split Rotate-then-Translate keyframes in the `usd-core` exporter.