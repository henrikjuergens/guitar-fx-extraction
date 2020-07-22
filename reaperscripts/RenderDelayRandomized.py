from pathlib import Path
import os
import random
import pickle

DATA_PATH = Path('D:/Henrik/Eigene Aufnahmen/Samples/Masterarbeit')
RPR_RENDER_PATH = Path('D:/Henrik/Eigene Aufnahmen/ReaperProjekte/DelayRandomized')
DIST_FX_SLOT = 0


def randomize_effect(CurTr):
  rand_param = [0, 3]
  param_vals = [random.uniform(0.0, 1.0) for _ in range(len(rand_param))]
  param_vals[0] = 0.25*param_vals[0] # Set Dry to -6db max
  param_vals[1] = 0.1*param_vals[1] # Set Length to 1s max
  for index, param in enumerate(rand_param):
    RPR_TrackFX_SetParam(CurTr, DIST_FX_SLOT, param, param_vals[index])
  return param_vals


def render_files(file_name):
  RPR_InsertMedia(file_name, 0)
  CurIt = RPR_GetMediaItem(0, 0)
  CurTr = RPR_GetMediaItem_Track(CurIt)
  RPR_Main_OnCommand(40108, 0) # Normalize
  param_vals = randomize_effect(CurTr)
  param_val_file = str(RPR_RENDER_PATH) + '\\' 'CurrentDir' + '\\' + file_name[:-4] + '-DlyRnd' + '.pickle'
  with open(param_val_file, 'wb') as handle:
    pickle.dump(param_vals, handle)
    
  RPR_Main_OnCommand(41824, 0) # Render
  RPR_DeleteTrackMediaItem(CurTr, CurIt)
  RPR_SetEditCurPos(0.0, False, False)


def process_files():
  RPR_PreventUIRefresh(1)
  directories = ['Gitarre monophon/Samples',
  'Gitarre polyphon/Samples']
  RPR_ShowConsoleMsg('\nStart\n')
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
      if effect_folder == 'NoFX':
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

