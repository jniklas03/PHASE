from phase import timelapse_pipeline
from test_paths import TIMELAPSE, TIMELAPSE_ACEGLU, SAVE_PATH
import pandas as pd 

ANTIBIOTIKA = r"C:\Users\jakub\Documents\Bachelorarbeit\Resources\Sources\02.12.2025"


def save_dishstates_to_excel(dish_states, save_path="dish_states.xlsx"):
    """
    Save dish_states to an Excel file.
    
    dish_states: list of DishState objects
    save_path: path to save the Excel file
    """
    # Prepare a list to hold all rows
    data_rows = []

    for idx, dish in enumerate(dish_states, start=1):
        for delta_t, count in dish.history:
            data_rows.append({
                "Dish": idx,
                "Time (hours)": delta_t,
                "Colony Count": count
            })

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Save to Excel
    df.to_excel(save_path, index=False)
    print(f"Dish states saved to {save_path}")

data = timelapse_pipeline(
    source=ANTIBIOTIKA, 
    save_intermediates=False,
    save_path=SAVE_PATH, 
    plot=True,
    use_masks=False,
    fine_buffer=0,
    n_to_stack=0
)

save_dishstates_to_excel(data, "02.12.xlsx")