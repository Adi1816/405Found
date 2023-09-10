import pandas as pd
from datetime import datetime

file_name = "recognized_faces.csv"
file_name_output = "Attendance.csv"

# Read the existing CSV file
df = pd.read_csv(file_name, sep="\t or ,")

df.drop_duplicates(subset=None, inplace=True)

# Create a new DataFrame with date and time entries
now = datetime.now()
date_time_df = pd.DataFrame({'Date': [now.strftime('%Y-%m-%d')],
                             'Time': [now.strftime('%H:%M:%S')]})

# Concatenate the new DataFrame with the existing DataFrame
df = pd.concat([date_time_df, df], axis=1)

# Write the results to a different file
df.to_csv(file_name_output, index=False)

print("")
print("")
print("")
print("Attendance File Created -- Check the Containing Folder")
