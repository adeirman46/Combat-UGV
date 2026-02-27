document.addEventListener('DOMContentLoaded', () => {
    
    // Grab inputs
    const inMass = document.getElementById('mass');
    const inVMax = document.getElementById('v_max');
    const inSlope = document.getElementById('slope');
    const inMotors = document.getElementById('motors');
    const inDuration = document.getElementById('duration');
    const inVAvg = document.getElementById('v_avg');
    const inCrr = document.getElementById('c_rr');
    const inPAux = document.getElementById('p_aux');

    const btnCalc = document.getElementById('btn-calculate');
    const btnExport = document.getElementById('btn-export');

    // Make calculation on startup
    calculate();

    btnCalc.addEventListener('click', calculate);

    function calculate() {
        const m = parseFloat(inMass.value);
        const v_max_kmh = parseFloat(inVMax.value);
        const slope = parseFloat(inSlope.value);
        const motors = parseInt(inMotors.value);
        const duration = parseFloat(inDuration.value);
        const v_avg_kmh = parseFloat(inVAvg.value);
        const C_rr = parseFloat(inCrr.value);
        const P_aux = parseFloat(inPAux.value); // in kW
        
        const g = 9.81;

        // 1. Physics of Slope Climbing
        // slope angle
        const theta_rad = Math.atan(slope / 100);
        const theta_deg = theta_rad * (180 / Math.PI);

        // Forces
        const F_g = m * g * Math.sin(theta_rad);
        const F_rr = m * g * Math.cos(theta_rad) * C_rr;
        const F_total = F_g + F_rr;

        // 2. Motor Power (Peak based on v_max on slope)
        const v_max_ms = v_max_kmh / 3.6;
        const P_motion_W = F_total * v_max_ms;
        const P_motion_kW = P_motion_W / 1000;
        const P_motor_kW = P_motion_kW / motors;

        // 3. Battery Capacity & Endurance (using v_avg on flat terrain)
        const v_avg_ms = v_avg_kmh / 3.6;
        // P_avg_motion = (m * g * C_rr) * v_avg (Assuming flat terrain where F_g is 0)
        const F_rr_flat = m * g * C_rr; // cos(0) = 1
        const P_avg_motion_W = F_rr_flat * v_avg_ms;
        const P_avg_motion_kW = P_avg_motion_W / 1000;
        
        const P_total_avg_kW = P_avg_motion_kW + P_aux;
        const E_batt_kWh = P_total_avg_kW * duration;

        // 4. Max Distance
        const D_max = v_avg_kmh * duration;

        // Update UI
        updateUI('out-theta', theta_deg, 2);
        updateUI('out-fg', F_g, 2);
        updateUI('out-frr', F_rr, 2);
        updateUI('out-ftotal', F_total, 2);

        updateUI('out-pmotion', P_motion_kW, 2);
        updateUI('out-pmotor', P_motor_kW, 2);

        updateUI('out-motor-count', motors, 0);
        updateUI('out-slope-disp', slope, 0);
        updateUI('out-vmax-disp', v_max_kmh, 0);

        updateUI('out-pavg', P_avg_motion_kW, 3);
        updateUI('out-vavg-disp', v_avg_kmh, 0);
        updateUI('out-ptot-avg', P_total_avg_kW, 3);
        updateUI('out-paux-disp', P_aux, 1);

        updateUI('out-ebatt', E_batt_kWh, 3);
        updateUI('out-dur-disp', duration, 0);

        updateUI('sum-pmotion', P_motion_kW, 2);
        updateUI('sum-pmotor', P_motor_kW, 2);
        updateUI('sum-ebatt', E_batt_kWh, 2);
        updateUI('sum-dist', D_max, 1);
    }

    function updateUI(id, val, decimals) {
        document.getElementById(id).textContent = val.toFixed(decimals);
    }

    // Export PDF
    btnExport.addEventListener('click', () => {
        const element = document.getElementById('report-content');
        
        // Options for html2pdf
        const opt = {
            margin:       10,
            filename:     'EV_UGV_Calculation_Report.pdf',
            image:        { type: 'jpeg', quality: 0.98 },
            html2canvas:  { scale: 2 },
            jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };
        
        // Create and save PDF
        html2pdf().set(opt).from(element).save();
    });

});
