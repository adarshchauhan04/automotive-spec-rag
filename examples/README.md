# Sample Queries and Outputs

This directory contains sample extraction outputs from `sample-service-manual 1.pdf` (Ford F-150 Workshop Manual) showing the queries, retrieved manual excerpts, and generated JSON/CSV outputs.

---

## Sample Queries

### Example 1: Fastener Torque Specification (Front & Rear Disc Brakes)
- **Natural Language Query:**  
  `"What is the torque specification for the brake caliper bolts?"`
- **Manual Section:**  
  `SECTION 206-03: Front Disc Brake` (Page 636) & `SECTION 206-04: Rear Disc Brake` (Page 652)
- **Retrieved Excerpt:**
  ```text
  | Description | Nm | lb-ft | lb-in |
  | Brake caliper flow bolt | 35 | 26 | — |
  | Brake caliper guide pin bolts | 37 | 27 | — |
  | Brake caliper anchor plate bolts | 250 | 184 | — |
  | Brake caliper support bracket bolts | 150 | 111 | — |
  ```
- **Validated JSON Output:** [example_1_brake_torque.json](./example_1_brake_torque.json)
- **Validated CSV Output:** [example_1_brake_torque.csv](./example_1_brake_torque.csv)

---

### Example 2: Fastener Torque Specification (Wheel Stud Nuts)
- **Natural Language Query:**  
  `"What is the torque specification for the wheel nuts?"`
- **Manual Section:**  
  `SECTION 204-04: Wheels and Tires` (Page 197)
- **Retrieved Excerpt:**
  ```text
  NOTE: The wheel nut torque specification is for clean, dry wheel stud and wheel nut threads.
  Install the 6 or 7 wheel nuts by hand.
  Tighten to 204 Nm (150 lb-ft).
  ```
- **Validated JSON Output:** [example_2_wheel_nut_torque.json](./example_2_wheel_nut_torque.json)
- **Validated CSV Output:** [example_2_wheel_nut_torque.csv](./example_2_wheel_nut_torque.csv)

---

### Example 3: Fluid Capacity Specification (DOT 3 Brake Fluid)
- **Natural Language Query:**  
  `"What is the fill capacity for High Performance DOT 3 Brake Fluid?"`
- **Manual Section:**  
  `SECTION 206-00: Brake System - General Information` (Pages 599, 636, 696)
- **Retrieved Excerpt:**
  ```text
  Motorcraft® High Performance DOT 3 Motor Vehicle Brake Fluid (US) /
  Motorcraft® Brake Fluid - Heavy Duty DOT 3 (Canada) PM-1-C (US); CPM-1-C (Canada)
  | WSS-M6C62-A or WSS-M6C65-A1 | 1,420 ml (3.0 pt) |
  ```
- **Validated JSON Output:** [example_3_brake_fluid_capacity.json](./example_3_brake_fluid_capacity.json)
- **Validated CSV Output:** [example_3_brake_fluid_capacity.csv](./example_3_brake_fluid_capacity.csv)

---

### Example 4: Anti-Hallucination & Out-of-Scope Rejection
- **Natural Language Query:**  
  `"What is the recommended tire inflation pressure for an Airbus A380 or a Tesla Model 3?"`
- **Result:**  
  `[]` (Empty list)
- **Design Guarantee:**  
  The system is strictly prompt-engineered to forbid hallucinating values or guessing from pretraining weights. When a query references non-existent entities or ungrounded specifications, the model returns an empty list `[]` instead of inventing answers.
- **Record:** [example_4_anti_hallucination.json](./example_4_anti_hallucination.json)

---

## Schema Enforcement

Every specification conforms to the `VehicleSpecification` Pydantic model:

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `component` | `string` | Specific automotive part or assembly | `"Brake caliper flow bolt"` |
| `spec_type` | `string` | Nature of measurement | `"Torque"`, `"Fluid Capacity"` |
| `value` | `string` | Numeric magnitude or range | `"35"`, `"1420"` |
| `unit` | `string` | Standard engineering unit | `"Nm"`, `"lb-ft"`, `"ml"` |
| `page_number` | `integer` (optional) | Source PDF manual page (1-indexed) | `636` |
| `source_excerpt` | `string` (optional) | Exact quote from technical table/text | `"Brake caliper flow bolt \| 35 \| 26 \| — \|"` |
