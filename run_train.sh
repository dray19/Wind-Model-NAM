set -euo pipefail

# This script:
# - validates the input (YYYYMMDD)
# - computes first_of_month (keeps your existing behavior of appending 06)
# - iterates a list of model directories
# - runs each model's run.sh with the computed date
# - updates the forecast config with change_fcst_config.py
# - fails fast and prints helpful messages

if [[ "${#}" -ne 1 ]]; then
  echo "Usage: $0 YYYYMMDD" >&2
  exit 2
fi

date_arg1="$1"

# Validate YYYYMMDD (8 digits)
if ! [[ "${date_arg1}" =~ ^[0-9]{8}$ ]]; then
  echo "Error: date must be in YYYYMMDD format (8 digits)." >&2
  exit 2
fi

# Optional stricter validation: check month/day ranges using date -d if available
if command -v date >/dev/null 2>&1; then
  if ! date -d "${date_arg1:0:4}-${date_arg1:4:2}-${date_arg1:6:2}" >/dev/null 2>&1; then
    echo "Error: date '${date_arg1}' is not a valid calendar date." >&2
    exit 2
  fi
fi

# Preserve existing behavior: original script constructed first_of_month="${date_arg1}06"
# If you want first_of_month to be the first day of the month from the provided date_arg1,
# replace the next line accordingly (e.g., "${date_arg1:0:6}01").
first_of_month="${date_arg1}06"

models=(
  model_2y_2w_dc_bias_feat2
  model_5y_lin_bias_featAll
  model_4m_2w_dc_bias_featAll
  model_2y_lin_bias_featAll
  model_5y_lin_bias_feat2
  model_4m_2w_dc_bias_feat3
  model_5y_1y_featAll
  model_1y_4m_feat3
)

# Ensure helper script exists before starting
if [[ ! -f "change_fcst_config.py" ]]; then
  echo "Error: change_fcst_config.py not found in current directory." >&2
  exit 3
fi

echo "Starting run for date: ${first_of_month}"
for model in "${models[@]}"; do
  echo "---- Processing: ${model} ----"

  if [[ ! -d "${model}" ]]; then
    echo "Warning: directory '${model}' does not exist — skipping." >&2
    continue
  fi

  (
    cd "${model}"
    if [[ -f "run.sh" && ! -x "run.sh" ]]; then
      echo "Note: making run.sh executable in ${model}"
      chmod +x run.sh
    fi

    if [[ -x "run.sh" ]]; then
      echo "Running ${model}/run.sh ${first_of_month}"
      ./run.sh "${first_of_month}"
    else
      echo "Warning: run.sh not found or not executable in ${model} — skipping run." >&2
    fi
  )

  if python change_fcst_config.py "Production/config.ini" "${model}" "${first_of_month}"; then
    echo "Updated Production/config.ini for ${model}"
  else
    echo "Error: change_fcst_config.py failed for ${model}" >&2
    exit 4
  fi
done

echo "All done."