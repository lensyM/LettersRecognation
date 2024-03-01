# Import libs
from Analyzer import VideoAnalyzer

if __name__ == '__main__':

    # Inicjalizacja obiektu klasy VideoAnalyzer
    vision_analyzer = VideoAnalyzer(wideo='conv-track.avi')

    # Wywołanie metody run() - główna pętla analizy obrazu
    vision_analyzer.run()

    # Generacja raportu
    vision_analyzer.create_raport(file_name='raport')
