import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

NUM_DAYS = 7
NUM_NURSES = 50
NUM_PARTICLES = 30
MAX_ITER = 100

# Definisi shift
MORNING = 0
AFTERNOON = 1
NIGHT = 2
REST = 3

# Definisi sertifikat
CERT_NONE = 0
CERT_BABY = 1
CERT_ICU = 2
CERT_DENTAL = 3

# Warna untuk visualisasi
SHIFT_COLORS = {
    MORNING: '#FFD700',    # Kuning untuk pagi
    AFTERNOON: '#FF8C00',  # Oranye untuk sore
    NIGHT: '#003366',      # Biru tua untuk malam
    REST: '#F0F0F0'        # Abu-abu untuk libur
}

class Nurse:
    def __init__(self, id, name, age, experience, certificates):
        self.id = id
        self.name = name
        self.age = age
        self.experience = experience  # dalam tahun
        self.certificates = certificates
        self.is_experienced = experience >= 2

class HospitalWard:
    def __init__(self, name, morning_needs, afternoon_needs, night_needs, requires_cert=None):
        self.name = name
        self.morning_needs = morning_needs
        self.afternoon_needs = afternoon_needs
        self.night_needs = night_needs
        self.requires_cert = requires_cert or CERT_NONE

# Daftar bangsal (disederhanakan)
wards = [
    HospitalWard("Penyakit Menular", 4, 4, 4),
    HospitalWard("Penyakit Dalam", 2, 2, 2),
    HospitalWard("ICU", 4, 4, 4, requires_cert=CERT_ICU),
    HospitalWard("Ibu Melahirkan", 4, 4, 4),
    HospitalWard("Bayi Baru Lahir", 8, 8, 8, requires_cert=CERT_BABY),
    HospitalWard("Klinik Umum", 2, 2, 0),
    HospitalWard("Klinik Gigi", 2, 2, 0, requires_cert=CERT_DENTAL),
    HospitalWard("IGD", 8, 8, 8)
]

# Generate perawat acak
nurses = []
for i in range(NUM_NURSES):
    exp = random.randint(0, 5)
    certs = []
    if random.random() < 0.2:
        certs.append(CERT_BABY)
    if random.random() < 0.2:
        certs.append(CERT_ICU)
    if random.random() < 0.1:
        certs.append(CERT_DENTAL)
    
    nurses.append(Nurse(i, f"Perawat_{i+1}", random.randint(22, 55), exp, certs))

def calculate_optimal_nurses():
    """Menghitung jumlah perawat optimal"""
    total_per_shift = defaultdict(int)
    
    for ward in wards:
        total_per_shift["morning"] += ward.morning_needs
        total_per_shift["afternoon"] += ward.afternoon_needs
        total_per_shift["night"] += ward.night_needs
    
    max_daily = total_per_shift["morning"] + total_per_shift["afternoon"] + total_per_shift["night"]
    optimal = int(np.ceil((max_daily * NUM_DAYS) / (5 * 3)))  # 3 shift per hari kerja
    
    return max(optimal, len(nurses))

optimal_nurses = calculate_optimal_nurses()
print(f"Jumlah perawat optimal: {optimal_nurses}")

class Particle:
    def __init__(self, num_nurses, num_days):
        self.position = []
        self.velocity = []
        self.best_position = []
        self.best_fitness = -np.inf
        
        # Inisialisasi posisi dan velocity
        for _ in range(num_nurses):
            nurse_schedule = []
            nurse_velocity = []
            rest_days = random.sample(range(num_days), 2)  # 2 hari libur
            
            for day in range(num_days):
                if day in rest_days:
                    nurse_schedule.append(REST)
                else:
                    # Hindari shift berturut-turut
                    prev_shift = nurse_schedule[-1] if day > 0 else None
                    if prev_shift is not None and prev_shift != REST:
                        available_shifts = [MORNING, AFTERNOON, NIGHT]
                        available_shifts.remove(prev_shift)
                        shift = random.choice(available_shifts)
                    else:
                        shift = random.choice([MORNING, AFTERNOON, NIGHT])
                    nurse_schedule.append(shift)
                
                nurse_velocity.append(random.uniform(-0.5, 0.5))
            
            self.position.append(nurse_schedule)
            self.velocity.append(nurse_velocity)
        
        self.position = np.array(self.position)
        self.velocity = np.array(self.velocity)
        self.best_position = np.copy(self.position)

def fitness(schedule, nurses, wards):
    """Fungsi fitness dengan 3 komponen penilaian"""
    score = 0
    
    # 1. Kebutuhan staf (40%)
    ward_scores = 0
    for day in range(NUM_DAYS):
        morning_count = defaultdict(int)
        afternoon_count = defaultdict(int)
        night_count = defaultdict(int)
        
        for nurse_idx, nurse_schedule in enumerate(schedule):
            shift = nurse_schedule[day]
            nurse = nurses[nurse_idx]
            
            if shift == MORNING:
                for ward in wards:
                    if ward.morning_needs > 0:
                        morning_count[ward.name] += 1
            elif shift == AFTERNOON:
                for ward in wards:
                    if ward.afternoon_needs > 0:
                        afternoon_count[ward.name] += 1
            elif shift == NIGHT:
                for ward in wards:
                    if ward.night_needs > 0:
                        night_count[ward.name] += 1
        
        for ward in wards:
            if ward.morning_needs > 0:
                ward_scores += min(1, morning_count[ward.name] / ward.morning_needs)
            if ward.afternoon_needs > 0:
                ward_scores += min(1, afternoon_count[ward.name] / ward.afternoon_needs)
            if ward.night_needs > 0:
                ward_scores += min(1, night_count[ward.name] / ward.night_needs)
    
    score += ward_scores / (NUM_DAYS * len(wards) * 3) * 40
    
    # 2. Aturan shift (30%)
    rule_score = 0
    for nurse_schedule in schedule:
        for i in range(1, NUM_DAYS):
            prev = nurse_schedule[i-1]
            curr = nurse_schedule[i]
            if prev != REST and curr != REST and prev == curr:
                rule_score -= 5
        
        rest_days = sum(1 for s in nurse_schedule if s == REST)
        if rest_days == 2:
            rule_score += 10
        else:
            rule_score -= abs(rest_days - 2) * 5
    
    score += rule_score / len(nurses) * 30
    
    # 3. Kualifikasi (30%)
    qual_score = 0
    for day in range(NUM_DAYS):
        for ward in wards:
            if ward.requires_cert != CERT_NONE:
                cert_nurses = 0
                for nurse_idx, nurse_schedule in enumerate(schedule):
                    shift = nurse_schedule[day]
                    nurse = nurses[nurse_idx]
                    
                    if (shift == MORNING and ward.morning_needs > 0) or \
                       (shift == AFTERNOON and ward.afternoon_needs > 0) or \
                       (shift == NIGHT and ward.night_needs > 0):
                        if ward.requires_cert in nurse.certificates:
                            cert_nurses += 1
                
                qual_score += min(1, cert_nurses / max(1, 
                    ward.morning_needs + ward.afternoon_needs + ward.night_needs))
    
    score += qual_score / (NUM_DAYS * sum(1 for w in wards if w.requires_cert != CERT_NONE)) * 30
    
    return score

def plot_convergence(fitness_history):
    """Grafik konvergensi fitness per iterasi"""
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, 'b-', linewidth=2, label='Fitness Terbaik')
    plt.title('Konvergensi Algoritma PSO', fontsize=14)
    plt.xlabel('Iterasi', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_schedule(schedule, nurses):
    """Heatmap jadwal perawat"""
    plt.figure(figsize=(14, 8))
    
    # Buat colormap kustom
    cmap = ListedColormap([SHIFT_COLORS[s] for s in [MORNING, AFTERNOON, NIGHT, REST]])
    
    # Plot heatmap
    plt.imshow(schedule, cmap=cmap, aspect='auto', 
              vmin=0, vmax=3, interpolation='none')
    
    # Atur ticks dan labels
    plt.yticks(range(len(nurses)), [n.name for n in nurses], fontsize=8)
    plt.xticks(range(NUM_DAYS), [f'Hari {i+1}' for i in range(NUM_DAYS)])
    
    # Buat colorbar
    cbar = plt.colorbar(ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(['Pagi', 'Sore', 'Malam', 'Libur'])
    
    plt.title('Heatmap Jadwal Perawat', fontsize=14)
    plt.xlabel('Hari', fontsize=12)
    plt.ylabel('Perawat', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_shift_distribution(schedule):
    """Diagram batang distribusi shift per hari"""
    shift_names = ['Pagi', 'Sore', 'Malam', 'Libur']
    days = [f'Hari {i+1}' for i in range(NUM_DAYS)]
    
    # Hitung jumlah per shift per hari
    shift_counts = np.zeros((NUM_DAYS, 4))
    for day in range(NUM_DAYS):
        for shift in schedule[:, day]:
            shift_counts[day, shift] += 1
    
    # Plot stacked bar chart
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(NUM_DAYS)
    
    for i in range(4):
        plt.bar(days, shift_counts[:, i], bottom=bottom, 
               label=shift_names[i], color=SHIFT_COLORS[i])
        bottom += shift_counts[:, i]
    
    plt.title('Distribusi Shift per Hari', fontsize=14)
    plt.xlabel('Hari', fontsize=12)
    plt.ylabel('Jumlah Perawat', fontsize=12)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def pso_algorithm():
    # Parameter PSO
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    particles = [Particle(len(nurses), NUM_DAYS) for _ in range(NUM_PARTICLES)]
    gbest_position = None
    gbest_fitness = -np.inf
    fitness_history = []
    
    # Inisialisasi
    for p in particles:
        p_fitness = fitness(p.position, nurses, wards)
        if p_fitness > p.best_fitness:
            p.best_fitness = p_fitness
        
        if p_fitness > gbest_fitness:
            gbest_fitness = p_fitness
            gbest_position = np.copy(p.position)
    
    # Iterasi
    for iteration in range(MAX_ITER):
        for p in particles:
            # Update velocity
            r1 = random.random()
            r2 = random.random()
            p.velocity = w * p.velocity + \
                         c1 * r1 * (p.best_position - p.position) + \
                         c2 * r2 * (gbest_position - p.position)
            
            # Update position
            new_position = p.position + p.velocity
            
            # Konversi ke shift valid
            for i in range(len(nurses)):
                for j in range(NUM_DAYS):
                    if new_position[i][j] < 0.5:
                        new_position[i][j] = MORNING
                    elif 0.5 <= new_position[i][j] < 1.5:
                        new_position[i][j] = AFTERNOON
                    elif 1.5 <= new_position[i][j] < 2.5:
                        new_position[i][j] = NIGHT
                    else:
                        new_position[i][j] = REST
            
            # Update fitness
            new_fitness = fitness(new_position, nurses, wards)
            
            # Update pbest dan gbest
            if new_fitness > p.best_fitness:
                p.best_fitness = new_fitness
                p.best_position = np.copy(new_position)
            
            if new_fitness > gbest_fitness:
                gbest_fitness = new_fitness
                gbest_position = np.copy(new_position)
        
        fitness_history.append(gbest_fitness)
        print(f"Iterasi {iteration+1}/{MAX_ITER}, Fitness: {gbest_fitness:.2f}")
    
    return gbest_position, gbest_fitness, fitness_history

if __name__ == "__main__":
    best_schedule, best_fitness, fitness_history = pso_algorithm()
    
    print("\n=== HASIL AKHIR ===")
    print(f"Fitness Terbaik: {best_fitness:.2f}")
    
    # Visualisasi
    plot_convergence(fitness_history)
    plot_schedule(best_schedule, nurses)
    plot_shift_distribution(best_schedule)
    
    # Contoh jadwal 5 perawat pertama
    print("\nContoh Jadwal 5 Perawat Pertama:")
    for i in range(5):
        print(f"\n{nurses[i].name}:")
        for day in range(NUM_DAYS):
            shift = best_schedule[i][day]
            if shift == MORNING:
                print(f"  Hari {day+1}: Shift Pagi (06:00-14:00)")
            elif shift == AFTERNOON:
                print(f"  Hari {day+1}: Shift Sore (14:00-22:00)")
            elif shift == NIGHT:
                print(f"  Hari {day+1}: Shift Malam (22:00-06:00)")
            else:
                print(f"  Hari {day+1}: Libur")