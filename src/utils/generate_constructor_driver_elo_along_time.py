import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_driver_elo(csv_file_path, drivers=None, figsize=(14, 8), save_path=None):
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Determine which drivers to plot
    drivers_to_plot = drivers if drivers else df['Driver'].unique()
    
    # Plot each driver
    for driver in drivers_to_plot:
        driver_df = df[df['Driver'] == driver][['Date', 'DriverEloBefore']].drop_duplicates()
        ax.plot(driver_df['Date'], driver_df['DriverEloBefore'], 
                marker='o', markersize=3, label=driver, linewidth=2, alpha=0.7)
    
    # Format x-axis to show only years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Driver ELO Rating', fontsize=12)
    ax.set_title('Driver ELO Ratings Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig, ax


def plot_constructor_elo(csv_file_path, teams=None, figsize=(14, 8), save_path=None):

    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Determine which teams to plot
    teams_to_plot = teams if teams else df['Team'].unique()
    
    # Plot each team
    for team in teams_to_plot:
        team_df = df[df['Team'] == team][['Date', 'ConstructorEloBefore']].drop_duplicates()
        ax.plot(team_df['Date'], team_df['ConstructorEloBefore'], 
                marker='s', markersize=3, label=team, linewidth=2, alpha=0.7)
    
    # Format x-axis to show only years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Constructor ELO Rating', fontsize=12)
    ax.set_title('Constructor ELO Ratings Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

if __name__=="__main__":

    FILE_SOURCE = r'src\generated_data\races_data.csv'
    SAVE_FOLDER = r'src\statistics/'

    # Plot all drivers
    #fig, ax = plot_driver_elo(FILE_SOURCE,save_path=SAVE_FOLDER+'elo_ratings_drivers.png')
    #plt.show()

    # Plot specific drivers
    fig, ax = plot_driver_elo(
        FILE_SOURCE,
        drivers=['HAM', 'VER', 'LEC', 'NOR', 'PIA', 'SAINZ', 'RUS', 'ALO', 'STR', 'GAS', 'OCO', 'TSU', 'LAW', 'ALB', 'COL', 'MAG', 'HUL', 'BEA'],
        save_path=SAVE_FOLDER+'elo_ratings_2025_drivers.png'
    )
    plt.show()

    # Plot all constructors
    fig, ax = plot_constructor_elo(FILE_SOURCE,save_path=SAVE_FOLDER+'elo_ratings_constructors.png')
    plt.show()

    # Plot specific constructors
    fig, ax = plot_constructor_elo(
        FILE_SOURCE,
        teams=['Mercedes', 'Red Bull Racing', 'Ferrari', 'McLaren'],
        save_path=SAVE_FOLDER+'elo_ratings_specific_constructors.png'
    )
    plt.show()