"""
NBA Game Win Prediction using Machine Learning
Complete implementation with data collection, preprocessing, model training, AND live predictions
"""

# ============================================================================
# STEP 1: INSTALL REQUIRED LIBRARIES
# ============================================================================
# Run these in your terminal first:
# pip install nba_api pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NBA API
from nba_api.stats.endpoints import leaguegamefinder, teamgamelog
from nba_api.stats.static import teams

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# STEP 2: DATA COLLECTION
# ============================================================================

def get_nba_game_data(season='2023-24', max_games=500):
    """
    Fetch NBA game data using nba_api
    Season format: '2023-24' for 2023-2024 season
    """
    print("Fetching NBA game data...")
    
    # Get all NBA teams
    nba_teams = teams.get_teams()
    
    # Fetch games
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable='00'  # NBA
    )
    
    games = gamefinder.get_data_frames()[0]
    
    # Limit to max_games for faster processing
    games = games.head(max_games)
    
    print(f"Fetched {len(games)} game records")
    return games

# ============================================================================
# STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================

def preprocess_data(games_df):
    """
    Clean and engineer features from raw game data
    """
    print("\nPreprocessing data...")
    
    # Select relevant columns
    features = games_df[[
        'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL',
        'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 
        'REB', 'AST', 'STL', 'BLK', 'TOV', 
        'PLUS_MINUS'
    ]].copy()
    
    # Convert Win/Loss to binary (1 = Win, 0 = Loss)
    features['WIN'] = (features['WL'] == 'W').astype(int)
    
    # Determine home/away
    features['HOME'] = features['MATCHUP'].str.contains('vs.').astype(int)
    
    # Calculate rolling averages (last 5 games performance)
    features = features.sort_values(['TEAM_ID', 'GAME_DATE'])
    
    rolling_cols = ['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    
    for col in rolling_cols:
        features[f'{col}_ROLL5'] = (
            features.groupby('TEAM_ID')[col]
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )
    
    # Win streak feature
    features['WIN_STREAK'] = (
        features.groupby('TEAM_ID')['WIN']
        .transform(lambda x: x.rolling(window=5, min_periods=1).sum())
    )
    
    # Drop rows with missing values
    features = features.dropna()
    
    print(f"Data shape after preprocessing: {features.shape}")
    return features

# ============================================================================
# STEP 4: PREPARE TRAINING DATA
# ============================================================================

def prepare_training_data(features_df):
    """
    Split features and target, create train/test sets
    """
    print("\nPreparing training data...")
    
    # Define feature columns (exclude ID, date, and target columns)
    feature_cols = [
        'HOME', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 
        'REB', 'AST', 'STL', 'BLK', 'TOV',
        'PTS_ROLL5', 'FG_PCT_ROLL5', 'FT_PCT_ROLL5', 
        'FG3_PCT_ROLL5', 'REB_ROLL5', 'AST_ROLL5', 
        'STL_ROLL5', 'BLK_ROLL5', 'TOV_ROLL5', 'WIN_STREAK'
    ]
    
    X = features_df[feature_cols]
    y = features_df['WIN']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

# ============================================================================
# STEP 5: TRAIN MODELS
# ============================================================================

def train_models(X_train, X_test, y_train, y_test):
    """
    Train Random Forest and Logistic Regression models
    """
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    models = {}
    
    # Random Forest
    print("\n1. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # Logistic Regression
    print("2. Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    return models

# ============================================================================
# STEP 6: EVALUATE MODELS
# ============================================================================

def evaluate_models(models, X_test, y_test):
    """
    Evaluate and compare model performance
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
    
    return results

# ============================================================================
# STEP 7: VISUALIZE RESULTS
# ============================================================================

def visualize_results(results, models, X_test, y_test, feature_cols):
    """
    Create visualizations of model performance
    """
    print("\nGenerating visualizations...")
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    for i, (name, acc) in enumerate(results.items()):
        plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison.png")
    
    # 2. Feature Importance (Random Forest)
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10
        
        plt.figure(figsize=(10, 6))
        plt.barh([feature_cols[i] for i in indices], importances[indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 10 Most Important Features (Random Forest)', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: feature_importance.png")
    
    # 3. Confusion Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
        axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: confusion_matrices.png")
    
    plt.show()

# ============================================================================
# STEP 8: GET TEAM CURRENT STATS
# ============================================================================

def get_team_id(team_name):
    """
    Convert team name to team ID
    """
    nba_teams = teams.get_teams()
    
    for team in nba_teams:
        if team_name.lower() in team['full_name'].lower() or \
           team_name.lower() in team['nickname'].lower() or \
           team_name.lower() in team['abbreviation'].lower():
            return team['id'], team['full_name']
    
    return None, None

def get_team_recent_stats(team_name, season='2023-24', last_n_games=5):
    """
    Get recent statistics for a team
    """
    team_id, full_name = get_team_id(team_name)
    
    if team_id is None:
        print(f"Team '{team_name}' not found!")
        return None
    
    print(f"Fetching stats for {full_name}...")
    
    # Get recent games
    gamelog = teamgamelog.TeamGameLog(
        team_id=team_id,
        season=season
    )
    
    games = gamelog.get_data_frames()[0].head(last_n_games)
    
    if len(games) == 0:
        print(f"No recent games found for {full_name}")
        return None
    
    # Calculate averages
    stats = {
        'TEAM_ID': team_id,
        'TEAM_NAME': full_name,
        'PTS': games['PTS'].mean(),
        'FG_PCT': games['FG_PCT'].mean(),
        'FT_PCT': games['FT_PCT'].mean(),
        'FG3_PCT': games['FG3_PCT'].mean(),
        'REB': games['REB'].mean(),
        'AST': games['AST'].mean(),
        'STL': games['STL'].mean(),
        'BLK': games['BLK'].mean(),
        'TOV': games['TOV'].mean(),
        'PTS_ROLL5': games['PTS'].mean(),
        'FG_PCT_ROLL5': games['FG_PCT'].mean(),
        'FT_PCT_ROLL5': games['FT_PCT'].mean(),
        'FG3_PCT_ROLL5': games['FG3_PCT'].mean(),
        'REB_ROLL5': games['REB'].mean(),
        'AST_ROLL5': games['AST'].mean(),
        'STL_ROLL5': games['STL'].mean(),
        'BLK_ROLL5': games['BLK'].mean(),
        'TOV_ROLL5': games['TOV'].mean(),
        'WIN_STREAK': games['WL'].apply(lambda x: 1 if x == 'W' else 0).sum()
    }
    
    return stats

# ============================================================================
# STEP 9: PREDICT FUTURE GAME
# ============================================================================

def predict_game(team1_name, team2_name, home_team, model, scaler, feature_cols, season='2023-24'):
    """
    Predict the outcome of a game between two teams
    
    Args:
        team1_name: First team name
        team2_name: Second team name
        home_team: Which team is home (team1_name or team2_name)
        model: Trained model
        scaler: Fitted scaler
        feature_cols: List of feature column names
        season: NBA season
    """
    print("\n" + "="*60)
    print("PREDICTING GAME OUTCOME")
    print("="*60)
    
    # Get stats for both teams
    team1_stats = get_team_recent_stats(team1_name, season)
    team2_stats = get_team_recent_stats(team2_name, season)
    
    if team1_stats is None or team2_stats is None:
        return
    
    print(f"\nMatchup: {team1_stats['TEAM_NAME']} vs {team2_stats['TEAM_NAME']}")
    
    # Prepare features for both teams
    def prepare_features(stats, is_home):
        features = []
        for col in feature_cols:
            if col == 'HOME':
                features.append(1 if is_home else 0)
            else:
                features.append(stats[col])
        return features
    
    # Determine who's home
    team1_home = (home_team.lower() in team1_name.lower())
    team2_home = not team1_home
    
    # Create feature arrays
    team1_features = np.array([prepare_features(team1_stats, team1_home)])
    team2_features = np.array([prepare_features(team2_stats, team2_home)])
    
    # Scale features
    team1_scaled = scaler.transform(team1_features)
    team2_scaled = scaler.transform(team2_features)
    
    # Get predictions
    team1_pred = model.predict(team1_scaled)[0]
    team2_pred = model.predict(team2_scaled)[0]
    
    # Get probabilities (if model supports it)
    if hasattr(model, 'predict_proba'):
        team1_proba = model.predict_proba(team1_scaled)[0]
        team2_proba = model.predict_proba(team2_scaled)[0]
        
        team1_win_prob = team1_proba[1] * 100  # Probability of winning
        team2_win_prob = team2_proba[1] * 100
        
        # Normalize so they sum to 100%
        total = team1_win_prob + team2_win_prob
        team1_win_prob = (team1_win_prob / total) * 100
        team2_win_prob = (team2_win_prob / total) * 100
    else:
        # For models without predict_proba
        team1_win_prob = 50.0 if team1_pred == 1 else 30.0
        team2_win_prob = 100.0 - team1_win_prob
    
    # Display results
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    
    home_indicator1 = "[HOME] " if team1_home else "[AWAY] "
    home_indicator2 = "[HOME] " if team2_home else "[AWAY] "
    
    print(f"\n{home_indicator1}{team1_stats['TEAM_NAME']}")
    print(f"   Win Probability: {team1_win_prob:.1f}%")
    print(f"   Recent Avg: {team1_stats['PTS']:.1f} pts, {team1_stats['REB']:.1f} reb, {team1_stats['AST']:.1f} ast")
    print(f"   Win Streak (last 5): {int(team1_stats['WIN_STREAK'])} wins")
    
    print(f"\n{home_indicator2}{team2_stats['TEAM_NAME']}")
    print(f"   Win Probability: {team2_win_prob:.1f}%")
    print(f"   Recent Avg: {team2_stats['PTS']:.1f} pts, {team2_stats['REB']:.1f} reb, {team2_stats['AST']:.1f} ast")
    print(f"   Win Streak (last 5): {int(team2_stats['WIN_STREAK'])} wins")
    
    # Determine winner
    if team1_win_prob > team2_win_prob:
        winner = team1_stats['TEAM_NAME']
        confidence = team1_win_prob
    else:
        winner = team2_stats['TEAM_NAME']
        confidence = team2_win_prob
    
    print(f"\n{'='*60}")
    print(f"PREDICTION: {winner} WINS")
    print(f"Confidence: {confidence:.1f}%")
    print(f"{'='*60}\n")
    
    return {
        'team1': team1_stats['TEAM_NAME'],
        'team2': team2_stats['TEAM_NAME'],
        'team1_win_prob': team1_win_prob,
        'team2_win_prob': team2_win_prob,
        'predicted_winner': winner,
        'confidence': confidence
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main pipeline to run the entire NBA prediction project
    """
    print("="*60)
    print("NBA GAME WIN PREDICTION - MACHINE LEARNING PROJECT")
    print("="*60)
    
    try:
        # Step 1: Get data
        games_df = get_nba_game_data(season='2023-24', max_games=1000)
        
        # Step 2: Preprocess
        features_df = preprocess_data(games_df)
        
        # Step 3: Prepare training data
        X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_training_data(features_df)
        
        # Step 4: Train models
        models = train_models(X_train, X_test, y_train, y_test)
        
        # Step 5: Evaluate
        results = evaluate_models(models, X_test, y_test)
        
        # Step 6: Visualize
        visualize_results(results, models, X_test, y_test, feature_cols)
        
        # Step 7: Use the best model for predictions
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        
        print("\n" + "="*60)
        print(f"Using {best_model_name} for predictions (Accuracy: {results[best_model_name]:.2%})")
        print("="*60)
        
        # ========================================================
        # STEP 8: MAKE LIVE PREDICTIONS
        # ========================================================
        
        print("\n" + "="*60)
        print("LIVE GAME PREDICTIONS")
        print("="*60)
        
        # *** CHANGE THESE LINES TO TEST YOUR GAMES ***
        
        # TEST: Warriors vs Pelicans (Yesterday's game - Warriors won 124-106)
        print("\n\nTESTING: Warriors vs Pelicans (Actual result: Warriors won 124-106)")
        pred1 = predict_game(
            team1_name="Warriors",
            team2_name="Pelicans",
            home_team="Pelicans",
            model=best_model,
            scaler=scaler,
            feature_cols=feature_cols
        )
        
        # Add more predictions here!
        # Example: Bulls vs Nuggets (Tonight at 9pm)
        # pred2 = predict_game("Bulls", "Nuggets", "Bulls", best_model, scaler, feature_cols)
        
        print("\n" + "="*60)
        print("PROJECT COMPLETE!")
        print("="*60)
        print(f"\nModel Accuracy: {results[best_model_name]:.2%}")
        print("\nTo test more games, change the lines marked with '*** CHANGE THESE LINES ***'")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Make sure you have installed all required packages:")
        print("pip install nba_api pandas numpy scikit-learn matplotlib seaborn")

if __name__ == "__main__":
    main()