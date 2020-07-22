from pathlib import Path
import os
import random

DATA_PATH = Path('D:/Henrik/Eigene Aufnahmen/Samples/Masterarbeit')
RPR_RENDER_PATH = Path('D:/Henrik/Eigene Aufnahmen/ReaperProjekte/AmpSim')
AMP_FX_SLOT = 1


def randomize_effect(CurTr):
  for param in range(0, 15):
    RPR_TrackFX_SetParam(CurTr, AMP_FX_SLOT, param, random.uniform(0.0, 1.0))
  RPR_TrackFX_SetParam(CurTr, AMP_FX_SLOT, 10, random.uniform(0.05, 0.5)) # Set Gain
 
 
def render_files(file_name):
  RPR_InsertMedia(file_name, 0)
  CurIt = RPR_GetMediaItem(0, 0)
  CurTr = RPR_GetMediaItem_Track(CurIt)
  RPR_Main_OnCommand(40108, 0) # Normalize
  randomize_effect(CurTr)
  
  RPR_Main_OnCommand(41824, 0) # Render
  RPR_DeleteTrackMediaItem(CurTr, CurIt)
  RPR_SetEditCurPos(0.0, False, False)


def process_files():
  RPR_PreventUIRefresh(1)
  directories = ['Gitarre monophon/Samples', 'Gitarre monophon2/Samples',
  'Gitarre polyphon/Samples', 'Gitarre polyphon2/Samples']
  
  os.chdir(RPR_RENDER_PATH)
  os.mkdir('CurrentDir')
  os.chdir('CurrentDir')
  os.mkdir('Samples')
  os.chdir('Samples')
      
  
  for dr in directories:
    os.chdir(DATA_PATH)
    os.chdir(Path(dr))      
    RPR_ShowConsoleMsg(dr)
    for effect_folder in os.listdir(os.getcwd()):
      RPR_ShowConsoleMsg(effect_folder)
      os.chdir(RPR_RENDER_PATH)
      os.chdir(Path('CurrentDir/Samples'))
      os.mkdir('CurrentFX')
      
      os.chdir(DATA_PATH)
      os.chdir(dr)
      os.chdir(effect_folder)
      print(effect_folder)
      for file_name in os.listdir(os.getcwd()):
        if file_name.endswith(".wav"):
            render_files(file_name)
      
      os.chdir(RPR_RENDER_PATH)
      os.chdir(Path('CurrentDir/Samples'))
      os.rename('CurrentFX', effect_folder)
      
    os.chdir(RPR_RENDER_PATH)
    dr_name = dr.partition('/')[0]
    os.rename('CurrentDir', dr_name)
    os.mkdir('CurrentDir')
    os.chdir('CurrentDir')
    os.mkdir('Samples')
  
  os.chdir(DATA_PATH)
  RPR_PreventUIRefresh(-1)
process_files()
RPR_ShowConsoleMsg('\nDone\n')

