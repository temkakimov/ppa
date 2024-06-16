import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors  # Corrected import
from pyrogram import Client, filters
from sqlalchemy import create_engine, Column, Integer, Date
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Database setup
Base = declarative_base()

class DailyRating(Base):
    __tablename__ = 'daily_ratings'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    dopamine_consumption_rating = Column(Integer, nullable=True)
    date = Column(Date, default=lambda: datetime.now(timezone.utc).date())  # Only store the date

engine = create_engine('sqlite:///dopaminedb4.db', connect_args={'timeout': 30, 'check_same_thread': False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

@contextmanager
def session_scope():
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

# Dictionary to store user session states and ratings
user_sessions = {}

def store_rating(user_id, dopamine_consumption):
    session = Session()
    today_date = datetime.now().date()

    # Check if there's already an entry for today
    existing_entry = session.query(DailyRating).filter(
        DailyRating.user_id == user_id, 
        DailyRating.date == today_date
    ).first()

    if existing_entry:
        session.close()
        return "You have already submitted your rating for today."
    
    # If no entry exists for today, create a new one
    new_rating = DailyRating(
        user_id=user_id,
        date=today_date,
        dopamine_consumption_rating=dopamine_consumption
    )
    session.add(new_rating)
    session.commit()
    session.close()

    return "Your rating has been recorded."

def process_ratings(client, message, ratings):
    user_id = message.from_user.id
    dopamine_consumption = ratings['dopamine_consumption']

    session = Session()
    today_date = datetime.now().date()

    # Store the new rating
    new_rating = DailyRating(
        user_id=user_id,
        dopamine_consumption_rating=dopamine_consumption,
        date=today_date
    )
    session.add(new_rating)
    session.commit()
    session.close()

    client.send_message(chat_id=user_id, text="Your rating has been recorded.")

DAYS = ['Sun.', 'Mon.', 'Tues.', 'Wed.', 'Thurs.', 'Fri.', 'Sat.']
MONTHS = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'June', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.']

def date_heatmap(series, start=None, end=None, ax=None, **kwargs):
    dates = series.index.floor('D')
    group = series.groupby(dates)
    series = group.first()  # Assume one value per day

    start = pd.to_datetime(start or series.index.min())
    end = pd.to_datetime(end or series.index.max())
    end += np.timedelta64(1, 'D')

    start_sun = start - np.timedelta64((start.dayofweek + 1) % 7, 'D')
    end_sun = end + np.timedelta64(7 - end.dayofweek - 1, 'D')

    num_weeks = (end_sun - start_sun).days // 7
    heatmap = np.zeros((7, num_weeks))
    ticks = {}
    for week in range(num_weeks):
        for day in range(7):
            date = start_sun + np.timedelta64(7 * week + day, 'D')
            if date.day == 1:
                ticks[week] = MONTHS[date.month - 1]
            if date.dayofyear == 1:
                ticks[week] += f'\n{date.year}'
            if start <= date < end:
                heatmap[day, week] = series.get(date, 0)

    y = np.arange(8) - 0.5
    x = np.arange(num_weeks + 1) - 0.5

    ax = ax or plt.gca()
    mesh = ax.pcolormesh(x, y, heatmap, **kwargs)
    ax.invert_yaxis()

    ax.set_aspect('equal')

    ax.set_xticks(list(ticks.keys()))
    ax.set_xticklabels(list(ticks.values()))
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(DAYS)

    plt.sca(ax)
    plt.sci(mesh)

    return ax

def plot_history(ratings):
    data = {'Date': [rating.date for rating in ratings],
            'Value': [rating.dopamine_consumption_rating for rating in ratings]}
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)

    fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the size as needed

    # Define colormap with specified colors
    cmap = mcolors.ListedColormap(['white', 'lightgreen', 'lightcoral', 'darkred'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    date_heatmap(df['Value'], start=start_date, end=end_date, ax=ax, cmap=cmap, norm=norm, edgecolors='black')

    plot_filename = 'history_plot.png'
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)

    return plot_filename

def send_proactive_messages():
    with session_scope() as session:
        # Get a list of all distinct user IDs
        user_ids = session.query(DailyRating.user_id).distinct().all()

    for user_id_tuple in user_ids:
        user_id = user_id_tuple[0]
        try:
            app.send_message(chat_id=user_id, text="Good evening! Please rate your cheap dopamine consumption today (1: No, 2: A little, 3: Yes):")
        except Exception as e:
            print(f"Error sending message to user {user_id}: {e}")

# Initialize the scheduler
scheduler = BackgroundScheduler()

# Schedule the job to run every evening at 8 PM
scheduler.add_job(send_proactive_messages, CronTrigger(hour=20, minute=0))

# Start the scheduler
scheduler.start()

# Confidential
api_id = os.getenv('API_ID') 
api_hash = os.getenv('API_HASH')
bot_token = os.getenv('BOT_TOKEN')

app = Client("my_bot", api_id=api_id, api_hash=api_hash, bot_token=bot_token)

@app.on_message(filters.command("start"))
def start(client, message):
    user_id = message.from_user.id
    session = Session()
    today_date = datetime.now().date()

    # Check if there's already an entry for today
    existing_entry = session.query(DailyRating).filter(
        DailyRating.user_id == user_id, 
        DailyRating.date == today_date
    ).first()

    if existing_entry:
        session.close()
        client.send_message(chat_id=message.chat.id, text="You have already submitted your rating for today.")
    else:
        user_sessions[user_id] = {'state': 'dopamine_consumption', 'ratings': {}}
        client.send_message(chat_id=message.chat.id, text="Welcome! Did you indulge in consuming cheap dopamine today? (1: No, 2: A little, 3: Yes):")
        session.close()

# Help command
@app.on_message(filters.command("help"))
def help(client, message):
    help_text = """
This bot helps you track your cheap dopamine consumption. Cheap dopamine activities include excessive social media use, binge-watching, and other activities that provide instant gratification with little effort.

Commands:
- /start: Begin rating your dopamine consumption for today.
- /history: View your past ratings.
"""
    message.reply_text(help_text)

# History command
@app.on_message(filters.command("history"))
def history(client, message):
    user_id = message.from_user.id
    session = Session()
    ratings = session.query(DailyRating).filter_by(user_id=user_id).order_by(DailyRating.date.desc()).all()
    session.close()

    if not ratings:
        message.reply_text("No rating history found.")
        return

    # Plot and send the chart
    plot_filename = plot_history(ratings)
    message.reply_photo(photo=plot_filename)

@app.on_message(filters.private)
def handle_rating_input(client, message):
    user_id = message.from_user.id

    if user_id not in user_sessions:
        client.send_message(chat_id=user_id, text="Please start a session first using /start command.")
        return  # Ignore messages if the user hasn't started a rating session

    try:
        rating = int(message.text)
        if rating not in [1, 2, 3]:
            raise ValueError
    except ValueError:
        client.send_message(chat_id=user_id, text="Please enter a valid number: 1 (No), 2 (A little), or 3 (Yes).")
        return

    current_state = user_sessions[user_id]['state']
    user_sessions[user_id]['ratings'][current_state] = rating

    if current_state == 'dopamine_consumption':
        # All ratings collected, process them
        process_ratings(client, message, user_sessions[user_id]['ratings'])
        del user_sessions[user_id]  # Clean up the session

if __name__ == '__main__':
    app.run()