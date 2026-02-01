def print_validation_report(model_predictions_log, gold_standard_data, frame_tolerance=10):
    """
    Compares prediction log against ground truth with fuzzy matching for shot types.
    Levels:
    - EXACT:   forehand_flat == forehand_flat
    - PARTIAL: forehand_flat ~= forehand_openstands (Same Side)
    - WRONG:   forehand_flat != backhand_slice (Wrong Side)
    """
    if not gold_standard_data:
        print("\nNo Ground Truth data provided. Skipping validation report.")
        return

    print("\n" + "="*60)
    print(f"{'FINAL ACCURACY REPORT':^60}")
    print("="*60)
    
    # Counters
    exact_shots = 0
    partial_shots = 0
    correct_players = 0
    total_labels = len(gold_standard_data)

    for gt in gold_standard_data:
        # 1. Find matching prediction within tolerance
        match = None
        for pred in model_predictions_log:
            if abs(pred['frame'] - gt['frame']) <= frame_tolerance:
                match = pred
                break
        
        print(f"Frame {gt['frame']:<4}: ", end="")
        
        if match:
            # 2. Analyze Player ID
            player_ok = (match['player'] == gt['player'])
            if player_ok: correct_players += 1

            # 3. Analyze Shot Type (Fuzzy Match Logic)
            gt_shot = gt['shot'].lower()
            pred_shot = match['shot'].lower()
            
            shot_status = "WRONG"
            
            # Case A: Exact Match
            if gt_shot == pred_shot:
                shot_status = "EXACT"
                exact_shots += 1
            
            # Case B: Partial Match (Same "Family")
            else:
                # Define families
                is_forehand = "forehand" in gt_shot and "forehand" in pred_shot
                is_backhand = "backhand" in gt_shot and "backhand" in pred_shot
                is_serve    = ("serve" in gt_shot or "service" in gt_shot) and \
                              ("serve" in pred_shot or "service" in pred_shot)
                
                if is_forehand or is_backhand or is_serve:
                    shot_status = "PARTIAL"
                    partial_shots += 1
            
            # 4. Determine Icon & Print
            if shot_status == "EXACT" and player_ok:
                icon = "✅ PERFECT"
            elif shot_status == "PARTIAL" and player_ok:
                icon = "⚠️ GOOD SIDE"
            elif shot_status == "WRONG" and player_ok:
                icon = "❌ WRONG SHOT"
            elif not player_ok:
                icon = "❌ WRONG PLAYER"
            
            print(f"{icon:<15} | Expected: {gt['shot']} (P{gt['player']})")
            print(f"{'':<12} Found:    {match['shot']} (P{match['player']}) @ Frame {match['frame']}")
            
        else:
            print(f"{'❌ MISSED':<15} | Expected: {gt['shot']} (P{gt['player']})")

    # Final Stats
    print("-" * 60)
    print(f"Player ID Accuracy:     {correct_players}/{total_labels} ({correct_players/total_labels*100:.1f}%)")
    print(f"Shot Exact Matches:     {exact_shots}/{total_labels} ({exact_shots/total_labels*100:.1f}%)")
    print(f"Shot Partial Matches:   {partial_shots}/{total_labels} (Total Useful: {(exact_shots+partial_shots)/total_labels*100:.1f}%)")
    print("="*60 + "\n")