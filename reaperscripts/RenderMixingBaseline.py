from pathlib import Path
import os
import random
import pickle

DATA_PATH = Path('D:/Henrik/Eigene Aufnahmen/Samples/Masterarbeit')
RPR_RENDER_PATH = Path('D:/Henrik/Eigene Aufnahmen/ReaperProjekte/DistortionRandomized')


 
def render_files():
  # RPR_Main_OnCommand(40108, 0) # Normalize
  three_db_factor = 2**0.5  
  git_tr = RPR_GetTrack(0,0)
  bass_tr = RPR_GetTrack(0,1)
  RPR_SetMediaTrackInfo_Value(git_tr, "D_VOL", 0.25) # Set Volume to -12dbBFS
  RPR_SetMediaTrackInfo_Value(bass_tr, "D_VOL", 0.25/(three_db_factor**10)) # Set Volume to -42dbBFS
  git_tr_og_vol = RPR_GetTrackUIVolPan(git_tr, 0, 0)[2]
  bass_tr_og_vol = RPR_GetTrackUIVolPan(bass_tr, 0, 0)[2]  
  
  for i in range(0, 11):
    # RPR_ShowConsoleMsg(str(RPR_GetTrackUIVolPan(bass_tr, 0, 0)[2]) + '\n')
    RPR_SetMediaTrackInfo_Value(git_tr, "D_VOL", git_tr_og_vol/(three_db_factor**i)) # Set Volume -3dB
    RPR_SetMediaTrackInfo_Value(bass_tr, "D_VOL", bass_tr_og_vol*(three_db_factor**i)) # Set Volume +3dB
        
    RPR_Main_OnCommand(41824, 0) # Render
    
  # RPR_PreventUIRefresh(1)
  
  # RPR_PreventUIRefresh(-1)

render_files()
RPR_ShowConsoleMsg('\nDone\n')

