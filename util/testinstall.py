import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchaudio
import soundfile as sf
import soundcard as sc


def test_torch():
    print(f'Version: {torch.__version__}')
    x = torch.rand(5, 3)
    print(x)
    print(type(x))
    return isinstance(x, torch.Tensor)


def test_cuda():
    print(f'Cuda available: {torch.cuda.is_available()}')
    return True


def test_numpy():
    print(f'Version: {np.__version__}')
    a = np.array([[1, 2], [3, 4]])
    a *= 2
    print(a)
    return a[0][0] == 2


def test_torchvision():
    print(f'Version: {torchvision.__version__}')
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(transformations)
    return True


def test_soundfile():
    print(f'Version: {sf.__version__}')
    data, samplerate = sf.read('example.wav')
    print(type(data))
    print(samplerate)
    return samplerate == 8000


def test_soundcard():
    default_speaker = sc.default_speaker()
    samples, samplerate = sf.read('example.wav')
    print(samples.shape)
    samples = samples[0:10000]
    print(samples.shape)
    default_speaker.play(samples, samplerate=samplerate)
    return samplerate == 8000


def test_torchaudio():
    print(f'Backend: {torchaudio.get_audio_backend()}')
    metadata = torchaudio.info('example.wav')
    print(metadata)
    return metadata.sample_rate == 8000


def test_install(name, test_fn):
    res = False
    print()
    print(f'-------- {name} --------')
    try:
        res = test_fn()
        print('Test done')
    except BaseException as be:
        print(f'******** Exception when testing {name}', be)
    except RuntimeError as re:
        print(f'******** RuntimeError when testing {name}', re)

    return res


def run_tests():
    results = dict()

    def _test(name, fn):  results[name] = test_install(name, fn)

    _test('numpy', test_numpy)
    _test('torch', test_torch)
    _test('torchvision', test_torchvision)
    _test('soundfile', test_soundfile)
    _test('sound card', test_soundcard)
    _test('torchaudio', test_torchaudio)
    _test('cuda', test_cuda)

    print('-' * 80)
    print('All tests passed') if all(results.values()) else print('One or more tests failed')
    print()
    print(results)



if __name__ == '__main__':
    run_tests()