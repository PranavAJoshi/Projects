
# coding: utf-8

# # Part 1. Plot the gender degree data and make it look nice.

# In[10]:

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns 
sns.set()


# In[11]:

# Read the data into a pandas DataFrame.    
gender_degree_data = pd.read_csv("http://www.randalolson.com/wp-content/uploads/percent-bachelors-degrees-women-usa.csv") 


# In[12]:

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

# List of the majors (See the .csv file)
majors = ['Health Professions', 'Public Administration', 'Education', 'Psychology',    
          'Foreign Languages', 'English', 'Communications\nand Journalism',    
          'Art and Performance', 'Biology', 'Agriculture',    
          'Social Sciences and History', 'Business', 'Math and Statistics',    
          'Architecture', 'Physical Sciences', 'Computer Science',    
          'Engineering'] 


# In[13]:

# We're going to make this plot taller so we can see the data better
plt.figure(figsize=(12, 14)) # sets size of plot. Common sizes are (10, 7.5) and (12, 9)

# Insert code here to remove plot frame lines. Use the spines function
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)      #This is to remove the top frame line of plot.
ax.spines["bottom"].set_visible(False)   #This is to remove the bottom frame line of plot.
ax.spines["left"].set_visible(False)     #This is to remove the left frame line of plot.
ax.spines["right"].set_visible(False)    #This is to remove the right frame line of plot. 


# Insert code here to only show the tick marks on the left and bottom
ax.get_xaxis().tick_bottom();            #This is to only show the bottom tick mark.
ax.get_yaxis().tick_left();              #This is to only show the left tick mark.


# Insert code here to only show the plot in the limits of the data 0-90% and 1968-2014
plt.ylim(0, 90)                          #This will show the y-axis within the limits 0 to 90 percent.
plt.xlim(1968, 2014)                     #This will show the x-axis within the limits 1968 to 2014.

# Insert code here to change the tick marks on the left to be 0%, 10%, etc. Also make the fontsize=14
plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)    
plt.xticks(fontsize=14)                  #ytick will show the y-axis limits within the gap of 10 and increase the size.
                                         #xtick will only increase the size.
     
        
# Make dashed lines across the plot at each of 0%, 10%, ..., 90%. Hint: use plot to plot dashed lines with a for loop
for y in range(10, 91, 10):    
    plt.plot(range(1968, 2012), [y] * len(range(1968, 2012)), "--", lw=0.5, color="black", alpha=0.3) 
    
# Remove the tick marks with plt.tickparams  
plt.tick_params(bottom="off", left="off")  #This will remove the left and bottom tick marks of the plot.


for rank, column in enumerate(majors):    
    # Plot each line separately with its own color, using the Tableau 20 color set in order
    plt.plot(gender_degree_data.Year.values, gender_degree_data[column.replace("\n", " ")].values, lw=2.5,
             color=tableau20[rank])
    
    # Add a text label to the right end of every line. Most of the code below    
    # is adding specific offsets y position because some labels overlapped.    
    y_pos = gender_degree_data[column.replace("\n", " ")].values[-1] - 0.5 
    if column == "Foreign Languages":    
        y_pos += 0.8    
    elif column == "English":    
        y_pos -= 0.6    
    elif column == "Communications\nand Journalism":    
        y_pos += 0.8    
    elif column == "Art and Performance":    
        y_pos -= 0.4    
    elif column == "Agriculture":    
        y_pos += 1    
    elif column == "Business":    
        y_pos -= 0.8    
    elif column == "Math and Statistics":    
        y_pos += 0.8    
    elif column == "Architecture":    
        y_pos -= 0.6    
    elif column == "Computer Science":    
        y_pos += 0.8    
    elif column == "Engineering":    
        y_pos -= 0.4 
    
    #The above if-elif code will replace the string from its place as per the given value.
    
    
    # Notice that some of the text labels are overlapping. Can you use an if statement to make adjustments to these?
    plt.text(2011.5, y_pos, column, fontsize=14, color=tableau20[rank])
    
    
# Make the title of the plot "Percentage of Bachelor's degrees conferred to women in the USA by major (1970-2012)
# Does this look OK? Can you make it look better? Maybe bigger?
plt.title("Percentage of Bachelor's degrees conferred to women in the U.S.A.\n by major (1970-2012)", 
          fontsize=15, fontweight='bold')
    #This code will write a big, bold size title
    
    
# Add text at the bottom of the plot that cites the source and your name on two separate lines using one line of code
plt.text(0.32, -0.1, "Source:- http://www.randalolson.com/wp-content/uploads/percent-bachelors-degrees-women-usa.csv\n       By:- Pranav A Joshi",verticalalignment='bottom',  
         transform=ax.transAxes, color='green', fontsize=10)

# Show the plot
plt.show()


# # Part 2: Let's look at using some plots with error bars.

# # Plot with error bars of climate data.

# In[15]:

# This will allow us to compute some statistics
from scipy.stats import sem

# Take an array of numbers and produce a function that averages the n-values around each number to smooth the data
def sliding_mean(data_array, window=5):

    return array(new_list)

# Load the climate data
data_names = ['Date Number', 'Year', 'Month', 'Day', 'Day of Year', 'Anomaly']
climate_data = pd.read_table("http://berkeleyearth.lbl.gov/auto/Global/Complete_TAVG_daily.txt",names=data_names,
                            comment='%',delim_whitespace=True) 

# Let's see what the climate data look like; did they load correctly? (Comment this line out after you have verified that
# your data has loaded correctly)
climate_data


# In[16]:

# First, let's extract some statistics from our pandas DataFrame
years = climate_data.groupby("Year").Anomaly.mean().keys()
mean_anomalies = climate_data.groupby("Year").Anomaly.mean().values
sem_anomalies = climate_data.groupby("Year").Anomaly.apply(sem).values


# In[23]:

# Now, let's see a plot with the standard error
plt.figure(figsize=(12,9))             #This will resize the plot to (12,9)

ax = plt.subplot(111)
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False) 
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)  #This code will remove the top, bottom, left and right frame lines.

ax.get_xaxis().tick_bottom();
ax.get_yaxis().tick_left();            #This code will only show the bottom and left tick mark.

plt.ylim(-0.85, 1.090 )
plt.xlim(1880, 2014)                   #This code will only show the bottom tick mark.

plt.ylabel('Anomalies', fontsize=12)
plt.xlabel('Years', fontsize=12)         #This code will label the y-axis and x-axis of plot and make the size bigger.

plt.title("Plot with error bars of climate data by major (1880-2014)", fontsize=12, fontweight='bold')
                                       #This code is to write a title in bold with size 12.
    
plt.fill_between(years,mean_anomalies - sem_anomalies,mean_anomalies + sem_anomalies, color = "#3F5D7D")
plt.plot(years, mean_anomalies, lw=1)


# # Part 3: For our third visualization, let's create some histograms of the climate data.

# In[24]:

# Plot a histogram of the anomalies data
for a in range(1, 11):
    plt.figure(figsize=(8,5))               #This will resize the plot to (8,5)

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False) 
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)   #This code will remove the top, bottom, left and right frame lines.

    ax.get_xaxis().tick_bottom();
    ax.get_yaxis().tick_left();             #This code will only show the bottom and left tick mark.
    
    plt.ylabel('Number of Occurences', fontsize=12)
    plt.xlabel('Anomaly', fontsize=12)      #This code will label the y-axis and x-axis of plot.
    
    
    #This below code if-elif code will split the histogram in 10 parts in between the year 1880 to 2014
    # and will break the total of 49311 anamolies in 10 parts according to the year and graph
    if a == 1:
        #This code will make a graph for the year 1880-1893 and will show the anomalies from 0 to 4932
        plt.title("Histogram of Climate data for the year (1880-1893)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[0:4932], color="#3F5D7D", bins=100)            
    
    elif a == 2:
        #This code will make a graph for the year 1893-1906 and will show the anomalies from 4932 to 9861
        plt.title("Histogram of Climate data for the year (1893-1906)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[4932:9861], color="#3F5D7D", bins=100)
                                        
        
    elif a == 3:
        #This code will make a graph for the year 1906-1919 and will show the anomalies from 9861 to 14791
        plt.title("Histogram of Climate data for the year (1906-1919)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[9861:14791], color="#3F5D7D", bins=100)
        
    elif a == 4:
        #This code will make a graph for the year 1919-1932 and will show the anomalies from 14791 to 19721
        plt.title("Histogram of Climate data for the year (1919-1932)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[14791:19721], color="#3F5D7D", bins=100)   
        
    elif a == 5:
        #This code will make a graph for the year 1932-1945 and will show the anomalies from 19721 to 24651
        plt.title("Histogram of Climate data for the year (1932-1945)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[19721:24651], color="#3F5D7D", bins=100)
        
    elif a == 6:
        #This code will make a graph for the year 1945-1958 and will show the anomalies from 24651 to 29581
        plt.title("Histogram of Climate data for the year (1945-1958)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[24651:29581], color="#3F5D7D", bins=100)
        
    elif a == 7:
        #This code will make a graph for the year 1958-1972 and will show the anomalies from 29581 to 34511
        plt.title("Histogram of Climate data for the year (1958-1972)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[29581:34511], color="#3F5D7D", bins=100)
        
    elif a == 8:
        #This code will make a graph for the year 1973-1986 and will show the anomalies from 34511 to 39441
        plt.title("Histogram of Climate data for the year (1973-1986)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[34511:39441], color="#3F5D7D", bins=100) 
        
    elif a == 9:
        #This code will make a graph for the year 1987-2000 and will show the anomalies from 39441 to 44371
        plt.title("Histogram of Climate data for the year (1987-2000)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[39441:44371], color="#3F5D7D", bins=100)
        
    elif a == 10:
        #This code will make a graph for the year 2001-2014 and will show the anomalies from 44371 to 49311
        plt.title("Histogram of Climate data for the year (2001-2014)", fontsize=12, fontweight='bold')
        plt.hist(climate_data.Anomaly.values[44371:49311], color="#3F5D7D", bins=100)


# In[ ]:



