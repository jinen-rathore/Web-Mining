
### The World's Flight Network

### This is our main.py, which does the following:
### - Importing necessary packages and self-defined modules
### - Loading in the necessary datasets
### - Calling functions of the preprocessing module to adjust and merge data
### - Starting the main program and calling other modules

### In the program there are several options you can choose from:
### 0) Demos: These will give a quick view on the world's  flight network,  
###    or only the flights or airports
### 1) Inspect the used dataframes: See their columns, unique values, or look 
###    at the biggest airports, airlines and represented countries
### 2) Visualize with self-choosen parameters: Here you have the option to set 
###    different parameters. Do you want to select a certain amount of airlines
###    or airports, do you want a specific airport or airline, do you want 
###    a directed or undirected network, and/or do you want to adjust the 
###    their size according to their degree?
### 3) Compare airlines: In this option it is possible to select two airlines,
###    plot those on the worldmap and get their graph theory metrics
### 4) Exit program

### An extensive flowchart

### This program will answer the following research questions:
### 1) What is the biggest (most connected) airport on earth?
### 2) What is the biggest (most flightes) airline?
### 3) How do airlines differ from each other based on graph metrics?


# self-defined modules
try:
    import base_preprocessing as bpp
    import module_visualization_worldmap as worldmap
    import module_inspect_data as inspect
    import module_comparison_airlines as comp_air
    import module_clustering as clt
except Exception as err: 
    print("Something went wrong with one of the modules")
    print(err)


# define filenames of datasets that you want to load
# in this case a csv with all flight routes, a csv with geographical locations of airports,
# and a txt file (in csv format) with airline information
filename_routes = "D:/VIT/SEM6/CSE3024-WEBMINING/Jcomp/Flight-Network-Analysis/Dataset/routes.csv"
filename_airports = "D:/VIT/SEM6/CSE3024-WEBMINING/Jcomp/Flight-Network-Analysis/Dataset/airports-extended.csv"
filename_airlines = "D:/VIT/SEM6/CSE3024-WEBMINING/Jcomp/Flight-Network-Analysis/Dataset/airlines.csv"

# load flight routes data into dataframe
try:
    df_routes = bpp.load_data_routes_from_file(filename_routes)
except FileNotFoundError:
    print("file not found, please check filename_routes and current directory")
except Exception as err: 
    print("Something went wrong")
    print(err)

# load airports data into dataframe    
try:
    df_airports = bpp.load_data_airports_from_file(filename_airports)
except FileNotFoundError:
    print("file not found, please check filename_routes and current directory")
except Exception as err: 
    print("Something went wrong")
    print(err)   
    
# load airlines data into dataframe    
try:    
    df_airlines = bpp.load_data_airlines_from_file(filename_airlines)
except FileNotFoundError:
    print("file not found, please check filename_routes and current directory")
except Exception as err: 
    print("Something went wrong")
    print(err) 
    
    
# Preprocessing: merging and cleaning of dataframes

# left outer join of routes and airlines dataframes
df_airlines = df_airlines[1:]
df_airlines["airline ID"] = df_airlines["airline ID"].astype(int)
df_merge_airlines_info = bpp.left_merge_dataframes(df_routes, df_airlines, "airline ID")

# left outer join of routes and airports dataframes
df_merged = bpp.left_merge_dataframes(df_merge_airlines_info, df_airports, "source airport ID")

# reindex columns of dataframe
df_merged = df_merged.reindex(columns=["airline IATA code", "airline ID", "name airline", "country airline", "source airport", "source airport ID", "destination airport", "destination airport ID", "airport name", "airport city", "airport country", "latitude", "longitude"])

# cleaning of the merged dataframe
df_merged = bpp.clean_dataframe(df_merged)   


# Run program in loop until user chooses to exit

while True:   
# print options to user:
    choice = input("""What do you want to do?
    0\tSee demo visualization of the flight network.
    1\tInspect the dataframes               
    2\tVisualise flight network with self-chosen parameters.
    3\tCompare airlines.
    4\tClustering.
    5\tExit program.
    enter answer (0/1/2/3/4/5): """)
    
    # evaluate user choice and proceed accordingly
    if choice == "0": # see demo
        
        # start this program in the worldmap module with the merged dataframe as input
        worldmap.demo_program(df_merged)


    elif choice == "1": # Inspect data

        # start this program in the inspect module with the dataframes to inspect
        inspect.inspect_data(df_routes, df_airports, df_merged)
    
    
    elif choice == "2": # Visualize flight network
        
        # start this program in the worldmap module with the merged dataframe as input
        worldmap.visualisation_worldmap_program(df_merged)

            
    elif choice == "3": # Compare airlines
        
        # start this program in the compare airlines module with the merged dataframe as input
        comp_air.compare_airlines_program(df_merged)

    elif choice == "4": # Clustering 
        
        # start this program in the compare airlines module with the merged dataframe as input
        clt.clustering_program(df_merged)

    elif choice == "5": # Exit program 
        print("Thank you for using this program.")
        break 
    
    else:
        print("Choice not recognized. Try again.")
        
        
