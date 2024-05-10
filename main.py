import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import pygrib



def get_atmospheric_data(region, date):
    """
    Получение данных зондирования атмосферы из модели GFS.
    Возвращает массив данных с профилями температуры, влажности, давления и ветра.
    """
    try:
        # Определение URL-адреса для загрузки данных
        base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        date_str = date.strftime("%Y%m%d")
        url = f"{base_url}?file=gfs.t00z.pgrb2.0p25.f000&lev_2_m_above_ground=on&var_TCDC=on&var_TMP=on&var_RH=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon={region.split('/')[0]}&rightlon={region.split('/')[1]}&toplat={region.split('/')[3]}&bottomlat={region.split('/')[2]}&dir=%2Fgfs.{date_str}%2F00%2Fatmos"

        # Загрузка данных в формате GRIB
        grbs = pygrib.open(url)

        # Извлечение необходимых полей
        temperature = grbs.select(name='Temperature')[0].values  # Температура на высоте 2 м
        humidity = grbs.select(name='Relative humidity')[0].values  # Относительная влажность
        pressure = grbs.select(name='Pressure')[0].values  # Давление на уровне моря
        u_wind = grbs.select(name='u-component of wind')[0].values  # Компонента ветра по оси x на высоте 10 м
        v_wind = grbs.select(name='v-component of wind')[0].values  # Компонента ветра по оси y на высоте 10 м

        # Создание массива данных
        atmospheric_data = xr.Dataset({
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "u_wind": u_wind,
            "v_wind": v_wind
        })

        return atmospheric_data
    except Exception as e:
        print(f"Ошибка при получении данных зондирования атмосферы: {e}")
        return None

def analyze_atmospheric_data(atmospheric_data):
    """
    Анализ и обработка данных зондирования атмосферы.
    Возвращает интерполированные профили атмосферных параметров.
    """
    try:
        # Код для проверки качества данных, интерполяции и расчета параметров атмосферы
        temperature = atmospheric_data['temperature'].interp(height=np.linspace(0, 20000, 1000))
        humidity = atmospheric_data['humidity'].interp(height=np.linspace(0, 20000, 1000))
        pressure = atmospheric_data['pressure'].interp(height=np.linspace(0, 20000, 1000))
        wind_speed = np.sqrt(atmospheric_data['u_wind']**2 + atmospheric_data['v_wind']**2)
        wind_direction = np.arctan2(atmospheric_data['v_wind'], atmospheric_data['u_wind'])
        return temperature, humidity, pressure, wind_speed, wind_direction
    except Exception as e:
        print(f"Ошибка при анализе данных зондирования атмосферы: {e}")
        return None, None, None, None, None

def simulate_balloon_ascent(mass_cargo, volume_balloon, lift_force, temperature, humidity, pressure, wind_speed, wind_direction):
    """
    Моделирование подъема аэростатной оболочки.
    Возвращает массив координат траектории движения оболочки.
    """
    try:
        # Код для расчета вертикальной и горизонтальной скоростей, интегрирования уравнений движения
        height = 0
        horizontal_position = 0
        trajectory = []
        dt = 0.1  # Интервал времени между точками в секундах
        while height < 20000:
            # Расчет параметров движения с учетом атмосферных условий
            vertical_speed = (lift_force - mass_cargo * 9.8) / (mass_cargo + volume_balloon * 1.225)
            horizontal_speed = wind_speed * np.cos(wind_direction)
            height += vertical_speed * dt
            horizontal_position += horizontal_speed * dt
            trajectory.append((horizontal_position, height))
        return np.array(trajectory)
    except Exception as e:
        print(f"Ошибка при моделировании подъема аэростатной оболочки: {e}")
        return None

def optimize_balloon_parameters(target_height, temperature, humidity, pressure, wind_speed, wind_direction):
    """
    Оптимизация параметров аэростатной оболочки для достижения целевой высоты.
    Возвращает оптимальные значения массы груза и объема оболочки.
    """
    try:
        # Код для поиска оптимальных параметров оболочки
        mass_cargo = 2  # Начальное значение массы груза
        volume_balloon = 10  # Начальное значение объема оболочки
        lift_force = volume_balloon * (1.225 - 1.2) * 9.8
        trajectory = simulate_balloon_ascent(mass_cargo, volume_balloon, lift_force, temperature, humidity, pressure, wind_speed, wind_direction)
        if trajectory is None:
            return None, None
        max_height = np.max(trajectory[:, 1])
        while abs(max_height - target_height) > 100:
            if max_height < target_height:
                volume_balloon += 1
            else:
                mass_cargo -= 0.1
            lift_force = volume_balloon * (1.225 - 1.2) * 9.8
            trajectory = simulate_balloon_ascent(mass_cargo, volume_balloon, lift_force, temperature, humidity, pressure, wind_speed, wind_direction)
            if trajectory is None:
                return None, None
            max_height = np.max(trajectory[:, 1])
        return mass_cargo, volume_balloon
    except Exception as e:
        print(f"Ошибка при оптимизации параметров аэростатной оболочки: {e}")
        return None, None

def visualize_trajectory(trajectory, start_time, time_step):
    """
    Визуализация траектории движения аэростатной оболочки.
    """
    try:
        # Горизонтальная траектория с привязкой к карте
        fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax1.add_feature(cfeature.COASTLINE)
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.set_extent([np.min(trajectory[:, 0]), np.max(trajectory[:, 0]),
                       np.min(trajectory[:, 1]), np.max(trajectory[:, 1])], crs=ccrs.PlateCarree())
        ax1.plot(trajectory[:, 0], trajectory[:, 1], transform=ccrs.PlateCarree())

        # Добавление меток времени на график
        time = start_time
        for i, point in enumerate(trajectory):
            if i % int(time_step / 0.1) == 0:
                ax1.text(point[0], point[1], f"{time.strftime('%H:%M')}", transform=ccrs.PlateCarree(), fontsize=8)
                time += datetime.timedelta(seconds=time_step)

        ax1.set_title('Горизонтальная траектория движения аэростата')
        ax1.set_xlabel('Долгота, град.')
        ax1.set_ylabel('Широта, град.')

        # Вертикальная траектория в зависимости от времени
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        time = start_time
        ax2.plot(np.arange(0, len(trajectory) * time_step, time_step) / 60, trajectory[:, 1])
        ax2.set_title('Вертикальная траектория движения аэростата')
        ax2.set_xlabel('Время, мин')
        ax2.set_ylabel('Высота, м')

        plt.show()
    except Exception as e:
        print(f"Ошибка при визуализации траектории движения аэростата: {e}")

# Пользовательский ввод
target_height = float(input("Введите целевую высоту подъема (в метрах): "))
region = input("Введите регион для получения данных зондирования атмосферы (например, 45/10/50/20): ")
date_str = input("Введите дату для получения данных зондирования атмосферы (ГГГГ-ММ-ДД): ")
date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

# Получение и обработка данных зондирования атмосферы
atmospheric_data = get_atmospheric_data(region, date)
if atmospheric_data is None:
    print("Не удалось получить данные зондирования атмосферы. Завершение программы.")
    exit()

temperature, humidity, pressure, wind_speed, wind_direction = analyze_atmospheric_data(atmospheric_data)
if temperature is None or humidity is None or pressure is None or wind_speed is None or wind_direction is None:
    print("Не удалось обработать данные зондирования атмосферы. Завершение программы.")
    exit()

# Моделирование и оптимизация подъема аэростатной оболочки
mass_cargo, volume_balloon = optimize_balloon_parameters(target_height, temperature, humidity, pressure, wind_speed, wind_direction)
if mass_cargo is None or volume_balloon is None:
    print("Не удалось оптимизировать параметры аэростатной оболочки. Завершение программы.")
    exit()

trajectory = simulate_balloon_ascent(mass_cargo, volume_balloon, (volume_balloon * (1.225 - 1.2) * 9.8), temperature, humidity, pressure, wind_speed, wind_direction)
if trajectory is None:
    print("Не удалось смоделировать подъем аэростатной оболочки. Завершение программы.")
    exit()

# Визуализация траектории
start_time = datetime.datetime(date.year, date.month, date.day, 10, 0, 0)
time_step = 10  # Интервал времени между точками в секундах
visualize_trajectory(trajectory, start_time, time_step)
