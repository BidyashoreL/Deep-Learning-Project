import matplotlib.pyplot as plt

labels = ["Low", "Moderate", "High"]
areas = [137.29, 137.29, 141.45]
colors = ["green", "yellow", "red"]

plt.figure(figsize=(8,6))
plt.bar(labels, areas, color=colors)

plt.ylabel("Area (km²)")
plt.title("Flood Risk Area Distribution")

for i, v in enumerate(areas):
    plt.text(i, v + 2, f"{v:.2f}", ha="center")

plt.show()