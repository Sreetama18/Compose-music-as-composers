from pydub import AudioSegment

sound = AudioSegment.from_mp3("C:/Users/sopal/Documents/Dataset/SDBurman/AajKiRaatPiya.mp3")
sound.export("../output/AajKiRaatPiya.wav", format="wav")