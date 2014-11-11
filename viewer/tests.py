import random, numpy


def test_seek(video):
	random.seed()
	for i in range(4):
		frame_number = random.randint(0, min(video.total_frame_count() - 1, 200))
		frame_one = video.seek_frame(frame_number)
		video.reset()
		for i in range(frame_number):
			video.move2_next()
		frame_two = video.move2_next()
		if not numpy.array_equal(frame_one, frame_two):
			video.reset()
			return False
	video.reset()
	return True