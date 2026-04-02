import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# Load images
img1 = mpimg.imread("../06-viz-time/plots/Execution_Time_Comparison_MLP_Configuration_001.png")
img2 = mpimg.imread("../05-viz-accuracy/RK1_plots/000_cfg_RK1_MLP_0256-5.png")

# Create layout
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 1)

# Top-left
ax_middle_top = fig.add_subplot(gs[0, 0])
ax_middle_top.imshow(img1)
ax_middle_top.axis('off')

# Bottom-left
ax_middle_bottom = fig.add_subplot(gs[1, 0])
ax_middle_bottom.imshow(img2)
ax_middle_bottom.axis('off')

fig.text(
    0.5, 0.98,   # (x, y) in figure coordinates
    "SuperCap: C=10F, ESR=0.05Ω, Vhi=5V, Vlo=3V",
    ha='center',
    fontsize=10
)

plt.tight_layout()
plt.savefig("combined.png")
plt.show()

print(img1.shape)
print(img2.shape)