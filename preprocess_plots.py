import pandas as pd
import matplotlib.pyplot as plt
# from bs4 import BeautifulSoup as bs

df = pd.read_csv("C:/Users/vigne/PycharmProjects/FaceMask/data_table.csv")

# Explore number of faces per image
# Plot graph for number of faces per image for first 15 images
df1=df.groupby('Filename').count()
df1 = df1.reset_index()
ax = df1['ImgDim'][0:15].plot(kind = 'bar',color = ['r','g','b','y','m','c'])
ax.set_title("Num of faces per image file", fontsize=16)
ax.set_ylabel("Num of faces", fontsize=12);
ax.set_xlabel("Image file", fontsize=12);
plt.savefig("C:/Users/vigne/PycharmProjects/FaceMask/saved_figures/num_faces_15_images")
# plt.show()


# Explore the number of faces per class
df2=df.groupby('ClassStr').count()
df2.head()
ax = df2['ImgDim'].plot(kind = 'bar', color = ['m','g','b'])
ax.set_title("Num of faces per category class", fontsize=16)
ax.set_ylabel("Num of faces", fontsize=12);
ax.set_xlabel("Category class", fontsize=12);
plt.savefig("C:/Users/vigne/PycharmProjects/FaceMask/saved_figures/unbalanced_dataset")
# plt.show()

