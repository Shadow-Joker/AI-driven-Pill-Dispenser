/* ============================================================
   SMART PILL DISPENSER
   FRONTEND APPLICATION
============================================================ */


/* ============================================================
   DOM ELEMENTS
============================================================ */

const navItems =
    document.querySelectorAll(
        ".nav-item"
    );


const pages =
    document.querySelectorAll(
        ".page"
    );


const pageTitle =
    document.getElementById(
        "pageTitle"
    );


const pageSubtitle =
    document.getElementById(
        "pageSubtitle"
    );


const capturePatientButton =
    document.getElementById(
        "capturePatientButton"
    );


const retakePatientButton =
    document.getElementById(
        "retakePatientButton"
    );


const captureCanvas =
    document.getElementById(
        "captureCanvas"
    );


const registerPatientCamera =
    document.getElementById(
        "registerPatientCamera"
    );


const capturedPatientImage =
    document.getElementById(
        "capturedPatientImage"
    );


const capturePreviewContainer =
    document.getElementById(
        "capturePreviewContainer"
    );


const patientRegistrationForm =
    document.getElementById(
        "patientRegistrationForm"
    );


const patientsGrid =
    document.getElementById(
        "patientsGrid"
    );


const refreshPatientsButton =
    document.getElementById(
        "refreshPatientsButton"
    );


const stopMotorsButton =
    document.getElementById(
        "stopMotorsButton"
    );


const toast =
    document.getElementById(
        "toast"
    );


const toastMessage =
    document.getElementById(
        "toastMessage"
    );


const toastIcon =
    document.getElementById(
        "toastIcon"
    );

const medicationStateBadge =
    document.getElementById(
        "medicationStateBadge"
    );


const currentPatient =
    document.getElementById(
        "currentPatient"
    );


const currentPill =
    document.getElementById(
        "currentPill"
    );


const currentMedicationState =
    document.getElementById(
        "currentMedicationState"
    );


const systemTime =
    document.getElementById(
        "systemTime"
    );


const schedulerStatus =
    document.getElementById(
        "schedulerStatus"
    );


const nextMedication =
    document.getElementById(
        "nextMedication"
    );


const nextMedicationTime =
    document.getElementById(
        "nextMedicationTime"
    );


const scheduleList =
    document.getElementById(
        "scheduleList"
    );


const scheduleEmpty =
    document.getElementById(
        "scheduleEmpty"
    );


const medicationLogs =
    document.getElementById(
        "medicationLogs"
    );


const refreshLogsButton =
    document.getElementById(
        "refreshLogsButton"
    );


const scheduleEditModal =
    document.getElementById(
        "scheduleEditModal"
    );

const scheduleEditForm =
    document.getElementById(
        "scheduleEditForm"
    );

const scheduleEditBackdrop =
    document.getElementById(
        "scheduleEditBackdrop"
    );

const scheduleEditCloseButton =
    document.getElementById(
        "scheduleEditCloseButton"
    );

const scheduleEditCancelButton =
    document.getElementById(
        "scheduleEditCancelButton"
    );

let medicationSchedulesCache = [];


/* ============================================================
   APPLICATION STATE
============================================================ */

let capturedImageData = null;

let toastTimeout = null;


/* ============================================================
   PAGE INFORMATION
============================================================ */

const pageInformation = {

    dashboard: {

        title:
            "Dashboard",

        subtitle:
            "Live monitoring and pill dispensing control"

    },

    register: {

        title:
            "Register Patient",

        subtitle:
            "Capture and register a new patient"

    },

    patients: {

        title:
            "Registered Patients",

        subtitle:
            "View and manage patient records"

    }

};


/* ============================================================
   NAVIGATION
============================================================ */

navItems.forEach(
    function (item) {

        item.addEventListener(
            "click",
            function () {

                const pageName =
                    item.dataset.page;


                showPage(
                    pageName
                );

            }
        );

    }
);


/* ============================================================
   SHOW PAGE
============================================================ */

function showPage(
    pageName
) {

    navItems.forEach(
        function (item) {

            item.classList.remove(
                "active"
            );

        }
    );


    const activeNavItem =
        document.querySelector(
            `.nav-item[data-page="${pageName}"]`
        );


    if (
        activeNavItem
    ) {

        activeNavItem.classList.add(
            "active"
        );

    }


    pages.forEach(
        function (page) {

            page.classList.remove(
                "active"
            );

        }
    );


    const selectedPage =
        document.getElementById(
            `${pageName}Page`
        );


    if (
        selectedPage
    ) {

        selectedPage.classList.add(
            "active"
        );

    }


    if (
        pageInformation[pageName]
    ) {

        pageTitle.textContent =
            pageInformation[pageName].title;


        pageSubtitle.textContent =
            pageInformation[pageName].subtitle;

    }


    if (
        pageName === "patients"
    ) {

        loadPatients();

    }


    window.scrollTo(
        {
            top: 0,
            behavior: "smooth"
        }
    );

}


/* ============================================================
   SHOW TOAST
============================================================ */

function showToast(
    message,
    type = "success"
) {

    clearTimeout(
        toastTimeout
    );


    toast.className =
        `toast ${type}`;


    if (
        type === "success"
    ) {

        toastIcon.textContent =
            "✓";

    }

    else {

        toastIcon.textContent =
            "⚠";

    }


    toastMessage.textContent =
        message;


    requestAnimationFrame(
        function () {

            toast.classList.add(
                "show"
            );

        }
    );


    toastTimeout =
        setTimeout(
            function () {

                toast.classList.remove(
                    "show"
                );

            },
            4000
        );

}


/* ============================================================
   DETECTION STATUS
============================================================ */

async function updateDetectionStatus() {

    try {

        const response =
            await fetch(
                "/api/status"
            );


        if (
            !response.ok
        ) {

            throw new Error(
                "Status request failed"
            );

        }


        const status =
            await response.json();


        updateFaceStatus(
            status
        );


        updateMouthStatus(
            status
        );


        updateHandStatus(
            status
        );


        updateMedicationStatus(
            status
        );

        updatePillAiStatus(
            status
        );

        setSystemOnline(
            true
        );

    }

    catch (
        error
    ) {

        console.error(
            "Status error:",
            error
        );


        setSystemOnline(
            false
        );

    }

}

/* ============================================================
   PILL AI STATUS
   ============================================================ */

function updatePillAiStatus(status) {

    const pillAiStatus =
        document.getElementById(
            "pillAiStatus"
        );

    if (!pillAiStatus) {
        return;
    }

    const state =
        status.medication_state ||
        "MEDICATION_IDLE";

    let text = "AI Standby";

    if (
        state ===
        "MEDICATION_WAITING_FOR_PILL"
    ) {
        text = "Detecting Pill";
    }

    else if (
        state ===
        "MEDICATION_WAITING_FOR_PICKUP"
    ) {
        text = "Monitoring Pickup";
    }

    else {
        text = "AI Standby";
    }

    pillAiStatus.textContent = text;
}


/* ============================================================
   FACE STATUS
============================================================ */

function updateFaceStatus(
    status
) {

    const faceStatus =
        document.getElementById(
            "faceStatus"
        );


    const dashboardFaceText =
        document.getElementById(
            "dashboardFaceText"
        );


    let text =
        "No Face";

    const recognitionStatus =
        status.face_recognition_status || "NO FACE";

    if (
        recognitionStatus === "RECOGNIZED"
        && status.recognized_patient_name
    ) {

        const patientId =
            status.recognized_patient_id || "-";

        const score =
            Number(status.face_match_score || 0).toFixed(2);

        text =
            `${status.recognized_patient_name} (${patientId}) • ${score}`;

    }

    else if (
        recognitionStatus === "UNKNOWN"
    ) {

        text =
            "Unknown Patient";

    }

    else if (
        recognitionStatus === "MULTIPLE FACES"
    ) {

        text =
            "Multiple Faces";

    }

    else if (
        status.face_count === 1
    ) {

        text =
            "Face Detected • Identifying...";

    }

    else if (
        status.face_count > 1
    ) {

        text =
            `${status.face_count} Faces`;

    }


    faceStatus.textContent =
        text;


    dashboardFaceText.textContent =
        text;


    applyStatusColor(
        faceStatus,
        status.face_count === 1
    );


    applyStatusColor(
        dashboardFaceText,
        status.face_count === 1
    );

}


/* ============================================================
   MOUTH STATUS
============================================================ */

function updateMouthStatus(
    status
) {

    const mouthStatus =
        document.getElementById(
            "mouthStatus"
        );


    const dashboardMouthText =
        document.getElementById(
            "dashboardMouthText"
        );


    let text =
        "Closed";


    if (
        status.face_count === 0
    ) {

        text =
            "Waiting";

    }

    else if (
        status.mouth_open
    ) {

        text =
            "Open";

    }


    mouthStatus.textContent =
        text;


    dashboardMouthText.textContent =
        text;


    const positive =
        status.mouth_open === true;


    applyStatusColor(
        mouthStatus,
        positive
    );


    applyStatusColor(
        dashboardMouthText,
        positive
    );

}


/* ============================================================
   HAND STATUS
============================================================ */

function updateHandStatus(
    status
) {

    const handStatus =
        document.getElementById(
            "handStatus"
        );


    const dashboardHandText =
        document.getElementById(
            "dashboardHandText"
        );


    let text =
        "No Hands";


    if (
        status.hand_count > 0
    ) {

        if (
            status.hand_count === 1
        ) {

            text =
                "1 Hand";

        }

        else {

            text =
                `${status.hand_count} Hands`;

        }

    }


    handStatus.textContent =
        text;


    dashboardHandText.textContent =
        text;


    applyStatusColor(
        handStatus,
        status.hand_count > 0
    );


    applyStatusColor(
        dashboardHandText,
        status.hand_count > 0
    );

}

/* ============================================================
   MEDICATION STATUS
============================================================ */

function updateMedicationStatus(
    status
) {

    const state =
        status.medication_state ||
        "IDLE";


    const pill =
        status.medication_pill_number;


    const patient =
        status.medication_patient_id;


    const schedulerRunning =
        status.scheduler_running === true;


    if (
        currentPatient
    ) {

        const recognizedName =
            status.recognized_patient_name;

        const recognizedId =
            status.recognized_patient_id;

        if (recognizedName && recognizedId) {
            currentPatient.textContent =
                `${recognizedName} (${recognizedId})`;
        }
        else if (status.face_recognition_status === "MULTIPLE FACES") {
            currentPatient.textContent =
                "Multiple Faces";
        }
        else if (status.face_recognition_status === "UNKNOWN") {
            currentPatient.textContent =
                "Unknown Patient";
        }
        else {
            currentPatient.textContent =
                "No Recognized Patient";
        }

    }


    const scheduledPatientElement =
        document.getElementById(
            "scheduledPatient"
        );

    if (
        scheduledPatientElement
    ) {
        scheduledPatientElement.textContent =
            patient || "—";
    }


    if (
        currentPill
    ) {

        currentPill.textContent =
            pill
                ? `Pill ${pill}`
                : "—";

    }


    if (
        currentMedicationState
    ) {

        currentMedicationState.textContent =
            formatMedicationState(
                state
            );

    }


    if (
        medicationStateBadge
    ) {

        medicationStateBadge.textContent =
            formatMedicationState(
                state
            ).toUpperCase();


        medicationStateBadge.className =
            "status-badge " +
            getMedicationStateClass(
                state
            );

    }


    if (
        systemTime
    ) {

        systemTime.textContent =
            status.current_time ||
            "—";

    }


    if (
        schedulerStatus
    ) {

        schedulerStatus.textContent =
            schedulerRunning
                ? "ACTIVE"
                : "OFFLINE";


        schedulerStatus.className =
            schedulerRunning
                ? "status-badge success"
                : "status-badge warning";

    }

}


function formatMedicationState(
    state
) {

    const labels = {

        "MEDICATION_IDLE":
            "Idle",

        "WAITING_FOR_PATIENT":
            "Waiting for Patient",

        "MEDICATION_WAITING_FOR_PATIENT":
            "Waiting for Patient",

        "MEDICATION_DISPENSING":
            "Dispensing",

        "MEDICATION_WAITING_FOR_PILL":
            "Waiting for Pill",

        "MEDICATION_WAITING_FOR_HAND":
            "Waiting for Hand",

        "MEDICATION_WAITING_FOR_MOUTH":
            "Waiting for Mouth",

        "MEDICATION_TAKEN":
            "Medication Taken",

        "MEDICATION_TIMEOUT":
            "Timed Out"

    };


    return labels[state] ||
        String(
            state || "Idle"
        )
            .replace(
                /^MEDICATION_/,
                ""
            )
            .replace(
                /_/g,
                " "
            )
            .replace(
                /\b\w/g,
                function (
                    letter
                ) {

                    return letter.toUpperCase();

                }
            );

}


function getMedicationStateClass(
    state
) {

    if (
        state ===
        "MEDICATION_TAKEN"
    ) {

        return "success";

    }


    if (
        state ===
        "MEDICATION_TIMEOUT"
    ) {

        return "error";

    }


    if (
        state ===
        "MEDICATION_IDLE"
    ) {

        return "neutral";

    }


    return "active";

}

/* ============================================================
   LOAD MEDICATION SCHEDULE
============================================================ */

async function loadMedicationSchedule() {

    try {

        const response =
            await fetch(
                "/api/medication_schedule"
            );


        if (
            !response.ok
        ) {

            throw new Error(
                "Could not load medication schedule."
            );

        }


        const schedules =
            await response.json();

        medicationSchedulesCache =
            Array.isArray(schedules)
                ? schedules
                : [];

        renderMedicationSchedule(
            medicationSchedulesCache
        );


        updateNextMedication(
            schedules
        );

    }

    catch (
        error
    ) {

        console.error(
            "Schedule error:",
            error
        );

    }

}


function renderMedicationSchedule(
    schedules
) {

    if (
        !scheduleList ||
        !scheduleEmpty
    ) {

        return;

    }


    scheduleList.innerHTML =
        "";


    if (
        !schedules ||
        schedules.length === 0
    ) {

        scheduleEmpty.style.display =
            "block";

        return;

    }


    scheduleEmpty.style.display =
        "none";


    schedules
        .sort(
            function (
                a,
                b
            ) {

                return String(
                    a.time || ""
                ).localeCompare(
                    String(
                        b.time || ""
                    )
                );

            }
        )
        .forEach(
            function (
                schedule
            ) {

                const card =
                    document.createElement(
                        "div"
                    );


                card.className =
                    "schedule-item";


                card.innerHTML = `

                    <div class="schedule-time">

                        <strong>
                            ${escapeHtml(
                                schedule.time || "—"
                            )}
                        </strong>

                    </div>


                    <div class="schedule-details">

                        <strong>
                            Pill ${escapeHtml(
                                schedule.pill_number || "—"
                            )}
                        </strong>

                        <span>
                            Patient:
                            ${escapeHtml(
                                schedule.patient_id || "—"
                            )}
                        </span>

                        ${
                            schedule.medicine_name
                                ? `<span>${escapeHtml(schedule.medicine_name)}${schedule.quantity ? ` • Qty ${escapeHtml(schedule.quantity)}` : ""}</span>`
                                : ""
                        }

                    </div>


                    <div class="schedule-actions">

                        <span
                            class="
                                status-badge
                                ${
                                    schedule.enabled
                                        ? "success"
                                        : "neutral"
                                }
                            "
                        >

                            ${
                                schedule.enabled
                                    ? "ACTIVE"
                                    : "DISABLED"
                            }

                        </span>

                        <button
                            class="secondary-button schedule-edit-button"
                            data-schedule-id="${escapeAttribute(
                                schedule.schedule_id || ""
                            )}"
                        >
                            Edit
                        </button>

                        <button
                            class="danger-button schedule-delete-button"
                            data-schedule-id="${escapeAttribute(
                                schedule.schedule_id || ""
                            )}"
                        >
                            Delete
                        </button>

                    </div>

                `;


                scheduleList.appendChild(
                    card
                );

            }
        );


    document
        .querySelectorAll(
            ".schedule-delete-button"
        )
        .forEach(
            function (
                button
            ) {

                button.addEventListener(
                    "click",
                    function () {

                        deleteMedicationSchedule(
                            button.dataset.scheduleId
                        );

                    }
                );

            }
        );

    document
        .querySelectorAll(
            ".schedule-edit-button"
        )
        .forEach(
            function (
                button
            ) {

                button.addEventListener(
                    "click",
                    function () {
                        openScheduleEditModal(
                            button.dataset.scheduleId
                        );
                    }
                );

            }
        );

}


function openScheduleEditModal(
    scheduleId
) {

    const schedule = medicationSchedulesCache.find(
        function (item) {
            return String(item.schedule_id) === String(scheduleId);
        }
    );

    if (!schedule || !scheduleEditModal) {
        return;
    }

    document.getElementById("editScheduleId").value =
        schedule.schedule_id || "";
    document.getElementById("editScheduleMedicine").value =
        schedule.medicine_name || "";
    document.getElementById("editSchedulePill").value =
        String(schedule.pill_number || 1);
    document.getElementById("editScheduleQuantity").value =
        schedule.quantity || 1;
    document.getElementById("editScheduleTime").value =
        schedule.time || "";
    document.getElementById("editScheduleEnabled").value =
        schedule.enabled ? "true" : "false";
    document.getElementById("scheduleEditPatient").textContent =
        `Patient ${schedule.patient_id || "—"}`;

    scheduleEditModal.classList.remove("hidden");
}


function closeScheduleEditModal() {
    if (scheduleEditModal) {
        scheduleEditModal.classList.add("hidden");
    }
}


async function saveScheduleEdit(event) {

    event.preventDefault();

    const scheduleId =
        document.getElementById("editScheduleId").value;

    const payload = {
        medicine_name:
            document.getElementById("editScheduleMedicine").value.trim(),
        pill_number:
            Number(document.getElementById("editSchedulePill").value),
        quantity:
            Number(document.getElementById("editScheduleQuantity").value),
        time:
            document.getElementById("editScheduleTime").value,
        enabled:
            document.getElementById("editScheduleEnabled").value === "true"
    };

    if (!scheduleId || !payload.time) {
        showToast("Please provide a valid medication time.", "error");
        return;
    }

    try {
        const response = await fetch(
            `/api/medication_schedule/${encodeURIComponent(scheduleId)}`,
            {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            }
        );

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(
                result.message ||
                "Could not update medication schedule."
            );
        }

        closeScheduleEditModal();
        showToast("Medication schedule updated successfully.");
        await loadMedicationSchedule();
        await loadPatients();

    } catch (error) {
        showToast(error.message, "error");
    }
}


if (scheduleEditForm) {
    scheduleEditForm.addEventListener(
        "submit",
        saveScheduleEdit
    );
}

if (scheduleEditBackdrop) {
    scheduleEditBackdrop.addEventListener(
        "click",
        closeScheduleEditModal
    );
}

if (scheduleEditCloseButton) {
    scheduleEditCloseButton.addEventListener(
        "click",
        closeScheduleEditModal
    );
}

if (scheduleEditCancelButton) {
    scheduleEditCancelButton.addEventListener(
        "click",
        closeScheduleEditModal
    );
}


function updateNextMedication(
    schedules
) {

    if (
        !nextMedication ||
        !nextMedicationTime
    ) {

        return;

    }


    const activeSchedules =
        (schedules || [])
            .filter(
                function (
                    schedule
                ) {

                    return schedule.enabled;

                }
            );


    if (
        activeSchedules.length === 0
    ) {

        nextMedication.textContent =
            "No scheduled medication";

        nextMedicationTime.textContent =
            "—";

        return;

    }


    const now =
        new Date();


    let next =
        null;


    activeSchedules.forEach(
        function (
            schedule
        ) {

            const parts =
                String(
                    schedule.time
                ).split(
                    ":"
                );


            if (
                parts.length !== 2
            ) {

                return;

            }


            const candidate =
                new Date(
                    now
                );


            candidate.setHours(
                Number(parts[0]),
                Number(parts[1]),
                0,
                0
            );


            if (
                candidate <= now
            ) {

                candidate.setDate(
                    candidate.getDate() + 1
                );

            }


            if (
                !next ||
                candidate < next.date
            ) {

                next = {

                    date:
                        candidate,

                    schedule:
                        schedule

                };

            }

        }
    );


    if (!next) {

        return;

    }


    nextMedication.textContent =
        `Pill ${next.schedule.pill_number} • ` +
        `Patient ${next.schedule.patient_id || "—"}`;


    nextMedicationTime.textContent =
        next.schedule.time;

}

async function deleteMedicationSchedule(
    scheduleId
) {

    if (
        !scheduleId
    ) {

        return;

    }


    const confirmed =
        confirm(
            "Delete this medication schedule?"
        );


    if (
        !confirmed
    ) {

        return;

    }


    try {

        const response =
            await fetch(
                `/api/medication_schedule/${encodeURIComponent(scheduleId)}`,
                {
                    method:
                        "DELETE"
                }
            );


        const result =
            await response.json();


        if (
            !response.ok ||
            !result.success
        ) {

            throw new Error(
                result.message ||
                "Could not delete schedule."
            );

        }


        showToast(
            "Medication schedule deleted."
        );


        loadMedicationSchedule();

    }

    catch (
        error
    ) {

        showToast(
            error.message,
            "error"
        );

    }

}

/* ============================================================
   MEDICATION LOGS
============================================================ */

async function loadMedicationLogs() {

    if (
        !medicationLogs
    ) {

        return;

    }


    try {

        const response =
            await fetch(
                "/api/medication_logs?limit=50"
            );


        if (
            !response.ok
        ) {

            throw new Error(
                "Could not load medication logs."
            );

        }


        const logs =
            await response.json();


        renderMedicationLogs(
            logs
        );

    }

    catch (
        error
    ) {

        console.error(
            "Medication log error:",
            error
        );


        medicationLogs.innerHTML = `

            <div class="empty-state">

                <div>
                    ⚠️
                </div>

                <h3>
                    Could not load medication history
                </h3>

            </div>

        `;

    }

}


function renderMedicationLogs(
    logs
) {

    if (
        !medicationLogs
    ) {

        return;

    }


    if (
        !logs ||
        logs.length === 0
    ) {

        medicationLogs.innerHTML = `

            <div class="empty-state">

                <div>
                    📋
                </div>

                <h3>
                    No medication history
                </h3>

            </div>

        `;

        return;

    }


    medicationLogs.innerHTML = `

        <div class="logs-table">

            <div class="logs-row logs-header">

                <span>
                    Time
                </span>

                <span>
                    Patient
                </span>

                <span>
                    Pill
                </span>

                <span>
                    Event
                </span>

                <span>
                    Result
                </span>

            </div>


            ${logs.map(
                function (
                    log
                ) {

                    return `

                        <div class="logs-row">

                            <span>
                                ${escapeHtml(
                                    log.timestamp || "—"
                                )}
                            </span>

                            <span>
                                ${escapeHtml(
                                    log.patient_id || "—"
                                )}
                            </span>

                            <span>
                                ${
                                    log.pill_number
                                        ? `Pill ${escapeHtml(
                                            log.pill_number
                                        )}`
                                        : "—"
                                }
                            </span>

                            <span>
                                ${escapeHtml(
                                    log.event || "—"
                                )}
                            </span>

                            <span>

                                <span
                                    class="
                                        log-result
                                        ${
                                            log.result ===
                                            "SUCCESS"
                                                ? "success"
                                                : log.result ===
                                                  "FAILED"
                                                    ? "error"
                                                    : "neutral"
                                        }
                                    "
                                >

                                    ${escapeHtml(
                                        log.result || "—"
                                    )}

                                </span>

                            </span>

                        </div>

                    `;

                }
            ).join("")}

        </div>

    `;

}

if (
    refreshLogsButton
) {

    refreshLogsButton.addEventListener(
        "click",
        function () {

            loadMedicationLogs();

        }
    );

}


/* ============================================================
   STATUS COLORS
============================================================ */

function applyStatusColor(
    element,
    positive
) {

    if (
        !element
    ) {

        return;

    }


    if (
        positive
    ) {

        element.style.color =
            "#4ade80";

    }

    else {

        element.style.color =
            "#fbbf24";

    }

}


/* ============================================================
   SYSTEM ONLINE STATUS
============================================================ */

function setSystemOnline(
    online
) {

    const statusDot =
        document.getElementById(
            "systemStatusDot"
        );


    if (
        !statusDot
    ) {

        return;

    }


    if (
        online
    ) {

        statusDot.style.background =
            "#22c55e";


        statusDot.style.boxShadow =
            "0 0 10px rgba(34,197,94,0.8)";

    }

    else {

        statusDot.style.background =
            "#ef4444";


        statusDot.style.boxShadow =
            "0 0 10px rgba(239,68,68,0.8)";

    }

}


/* ============================================================
   CAPTURE PATIENT IMAGE
============================================================ */

capturePatientButton.addEventListener(
    "click",
    function () {

        if (
            !registerPatientCamera.complete ||
            registerPatientCamera.naturalWidth === 0
        ) {

            showToast(
                "Patient camera is not ready.",
                "error"
            );

            return;

        }


        const width =
            registerPatientCamera.naturalWidth;


        const height =
            registerPatientCamera.naturalHeight;


        captureCanvas.width =
            width;


        captureCanvas.height =
            height;


        const context =
            captureCanvas.getContext(
                "2d"
            );


        context.drawImage(
            registerPatientCamera,
            0,
            0,
            width,
            height
        );


        capturedImageData =
            captureCanvas.toDataURL(
                "image/jpeg",
                0.92
            );


        capturedPatientImage.src =
            capturedImageData;


        capturePreviewContainer.classList.remove(
            "hidden"
        );


        showToast(
            "Patient image captured successfully."
        );

    }
);


/* ============================================================
   RETAKE PATIENT IMAGE
============================================================ */

retakePatientButton.addEventListener(
    "click",
    function () {

        capturedImageData =
            null;


        capturedPatientImage.src =
            "";


        capturePreviewContainer.classList.add(
            "hidden"
        );


        showToast(
            "Ready to capture a new image."
        );

    }
);


/* ============================================================
   REGISTER PATIENT
============================================================ */

patientRegistrationForm.addEventListener(
    "submit",
    async function (
        event
    ) {

        event.preventDefault();


        const patientName =
            document.getElementById(
                "patientName"
            ).value.trim();


        const patientId =
            document.getElementById(
                "patientId"
            ).value.trim();


        const pill1Medicine =
            document.getElementById(
                "pill1Medicine"
            ).value.trim();


        const pill1Quantity =
            Number(
                document.getElementById(
                    "pill1Quantity"
                ).value
            );


        const pill1Time =
            document.getElementById(
                "pill1Time"
            ).value;


        const pill2Medicine =
            document.getElementById(
                "pill2Medicine"
            ).value.trim();


        const pill2Quantity =
            Number(
                document.getElementById(
                    "pill2Quantity"
                ).value
            );


        const pill2Time =
            document.getElementById(
                "pill2Time"
            ).value;


        if (
            !patientName
        ) {

            showToast(
                "Please enter the patient name.",
                "error"
            );

            return;

        }


        if (
            !patientId
        ) {

            showToast(
                "Please enter the patient ID.",
                "error"
            );

            return;

        }


        if (
            !capturedImageData
        ) {

            showToast(
                "Please capture the patient image first.",
                "error"
            );

            return;

        }


        const submitButton =
            patientRegistrationForm.querySelector(
                "button[type='submit']"
            );


        submitButton.disabled =
            true;


        submitButton.textContent =
            "Registering...";


        try {

            /*
             * Backend also generates and stores the
             * face embedding from this image.
             */

            const response =
                await fetch(
                    "/api/register_patient",
                    {

                        method:
                            "POST",

                        headers:
                            {
                                "Content-Type":
                                    "application/json"
                            },

                        body:
                            JSON.stringify(
                                {
                                    patient_id:
                                        patientId,

                                    name:
                                        patientName,

                                    image:
                                        capturedImageData,

                                    medications: [

                                            {
                                                pill_number:
                                                    1,

                                                medicine_name:
                                                    pill1Medicine,

                                                quantity:
                                                    pill1Quantity,

                                                time:
                                                    pill1Time
                                            },

                                            {
                                                pill_number:
                                                    2,

                                                medicine_name:
                                                    pill2Medicine,

                                                quantity:
                                                    pill2Quantity,

                                                time:
                                                    pill2Time
                                            }

                                        ]

                                }
                            )

                    }
                );


            const result =
                await response.json();


            if (
                !response.ok ||
                !result.success
            ) {

                throw new Error(
                    result.message ||
                    "Registration failed."
                );

            }


            showToast(
                "Patient registered successfully."
            );

            await loadMedicationSchedule();


            resetRegistrationForm();


            setTimeout(
                function () {

                    showPage(
                        "patients"
                    );

                },
                700
            );

        }

        catch (
            error
        ) {

            console.error(
                error
            );


            showToast(
                error.message,
                "error"
            );

        }

        finally {

            submitButton.disabled =
                false;


            submitButton.textContent =
                "✓ Register Patient";

        }

    }
);


/* ============================================================
   RESET REGISTRATION FORM
============================================================ */

function resetRegistrationForm() {

    patientRegistrationForm.reset();


    capturedImageData =
        null;


    capturedPatientImage.src =
        "";


    capturePreviewContainer.classList.add(
        "hidden"
    );

}


/* ============================================================
   LOAD PATIENTS
============================================================ */

async function loadPatients() {

    patientsGrid.innerHTML =

        `
        <div class="empty-state">

            <div>
                ⏳
            </div>

            <h3>
                Loading patients...
            </h3>

        </div>
        `;


    try {

        const response =
            await fetch(
                "/api/patients"
            );


        if (
            !response.ok
        ) {

            throw new Error(
                "Could not load patients."
            );

        }


        const patients =
            await response.json();


        renderPatients(
            patients
        );

    }

    catch (
        error
    ) {

        console.error(
            error
        );


        patientsGrid.innerHTML =

            `
            <div class="empty-state">

                <div>
                    ⚠️
                </div>

                <h3>
                    Could not load patients
                </h3>

            </div>
            `;

    }

}


/* ============================================================
   RENDER PATIENTS
============================================================ */

function renderPatients(
    patients
) {

    patientsGrid.innerHTML =
        "";


    if (
        !patients ||
        patients.length === 0
    ) {

        patientsGrid.innerHTML =

            `
            <div class="empty-state">

                <div>
                    👥
                </div>

                <h3>
                    No patients registered yet
                </h3>

            </div>
            `;

        return;

    }


    patients.forEach(
        function (
            patient
        ) {

            const imageSource =
                patient.image
                    ? `data:image/jpeg;base64,${patient.image}`
                    : "";


            const card =
                document.createElement(
                    "div"
                );


            card.className =
                "patient-card";


            card.innerHTML =

                `
                <img
                    class="patient-image"
                    src="${imageSource}"
                    alt="Patient Image"
                >

                <div class="patient-card-body">

                    <h3>
                        ${escapeHtml(patient.name || "Unknown")}
                    </h3>

                    <span class="patient-id">

                        ID:
                        ${escapeHtml(patient.patient_id || "-")}

                    </span>

                    <div class="patient-date">

                        Registered:
                        ${escapeHtml(patient.registered_at || "-")}

                    </div>

                    <div class="patient-schedules">
                        <strong>Medication Schedule</strong>
                        ${
                            medicationSchedulesCache
                                .filter(
                                    function (schedule) {
                                        return String(schedule.patient_id) === String(patient.patient_id);
                                    }
                                )
                                .sort(
                                    function (a, b) {
                                        return String(a.time || "").localeCompare(String(b.time || ""));
                                    }
                                )
                                .map(
                                    function (schedule) {
                                        return `
                                            <div class="patient-schedule-row">
                                                <div class="patient-schedule-info">
                                                    <strong>${escapeHtml(schedule.time || "—")} • Pill ${escapeHtml(schedule.pill_number || "—")}</strong>
                                                    <span>${escapeHtml(schedule.medicine_name || "Medication")}${schedule.quantity ? ` • Qty ${escapeHtml(schedule.quantity)}` : ""}</span>
                                                </div>
                                                <button class="secondary-button schedule-edit-button" data-schedule-id="${escapeAttribute(schedule.schedule_id || "")}">Edit</button>
                                            </div>
                                        `;
                                    }
                                )
                                .join("") || `<span>No medication schedule</span>`
                        }
                    </div>

                    <div class="patient-card-actions">

                        <button
                            class="danger-button delete-patient-button"
                            data-patient-id="${escapeAttribute(patient.patient_id || "")}"
                        >

                            🗑 Delete Patient

                        </button>

                    </div>

                </div>
                `;


            patientsGrid.appendChild(
                card
            );

        }
    );


    document
        .querySelectorAll(
            ".delete-patient-button"
        )
        .forEach(
            function (
                button
            ) {

                button.addEventListener(
                    "click",
                    function () {

                        const patientId =
                            button.dataset.patientId;


                        deletePatient(
                            patientId
                        );

                    }
                );

            }
        );

    document
        .querySelectorAll(
            ".patient-schedules .schedule-edit-button"
        )
        .forEach(
            function (button) {
                button.addEventListener(
                    "click",
                    function () {
                        openScheduleEditModal(
                            button.dataset.scheduleId
                        );
                    }
                );
            }
        );

}


/* ============================================================
   DELETE PATIENT
============================================================ */

async function deletePatient(
    patientId
) {

    if (
        !patientId
    ) {

        return;

    }


    const confirmed =
        confirm(
            `Delete patient ${patientId}?`
        );


    if (
        !confirmed
    ) {

        return;

    }


    try {

        const response =
            await fetch(
                `/api/patient/${encodeURIComponent(patientId)}`,
                {
                    method:
                        "DELETE"
                }
            );


        const result =
            await response.json();


        if (
            !response.ok ||
            !result.success
        ) {

            throw new Error(
                result.message ||
                "Could not delete patient."
            );

        }


        showToast(
            "Patient deleted successfully."
        );


        loadPatients();

    }

    catch (
        error
    ) {

        showToast(
            error.message,
            "error"
        );

    }

}


/* ============================================================
   REFRESH PATIENTS
============================================================ */

refreshPatientsButton.addEventListener(
    "click",
    function () {

        loadPatients();

    }
);


/* ============================================================
   DISPENSE BUTTONS
============================================================ */

document
    .querySelectorAll(
        ".dispense-button"
    )
    .forEach(
        function (
            button
        ) {

            button.addEventListener(
                "click",
                async function () {

                    const pillNumber =
                        Number(
                            button.dataset.pill
                        );


                    const durationInput =
                        document.getElementById(
                            `pill${pillNumber}Duration`
                        );


                    const speedInput =
                        document.getElementById(
                            `pill${pillNumber}Speed`
                        );


                    const duration =
                        Number(
                            durationInput.value
                        );


                    const speed =
                        Number(
                            speedInput.value
                        );


                    if (
                        !duration ||
                        duration <= 0
                    ) {

                        showToast(
                            "Enter a valid dispensing duration.",
                            "error"
                        );

                        return;

                    }


                    if (
                        !speed ||
                        speed < 1 ||
                        speed > 100
                    ) {

                        showToast(
                            "Motor speed must be between 1 and 100.",
                            "error"
                        );

                        return;

                    }


                    button.disabled =
                        true;


                    button.textContent =
                        "Dispensing...";


                    try {

                        const response =
                            await fetch(
                                "/api/dispense",
                                {

                                    method:
                                        "POST",

                                    headers:
                                        {
                                            "Content-Type":
                                                "application/json"
                                        },

                                    body:
                                        JSON.stringify(
                                            {
                                                pill_number:
                                                    pillNumber,

                                                duration:
                                                    duration,

                                                speed:
                                                    speed
                                            }
                                        )

                                }
                            );


                        const result =
                            await response.json();


                        if (
                            !response.ok ||
                            !result.success
                        ) {

                            throw new Error(
                                result.message ||
                                "Could not start dispensing."
                            );

                        }


                        showToast(
                            `Pill ${pillNumber} dispensing started.`
                        );

                    }

                    catch (
                        error
                    ) {

                        showToast(
                            error.message,
                            "error"
                        );

                    }

                    finally {

                        setTimeout(
                            function () {

                                button.disabled =
                                    false;


                                button.textContent =
                                    `💊 Dispense Pill ${pillNumber}`;

                            },
                            Math.max(
                                duration * 1000,
                                1000
                            )
                        );

                    }

                }
            );

        }
    );


/* ============================================================
   STOP ALL MOTORS
============================================================ */

stopMotorsButton.addEventListener(
    "click",
    async function () {

        try {

            const response =
                await fetch(
                    "/api/stop_motors",
                    {

                        method:
                            "POST"

                    }
                );


            const result =
                await response.json();


            if (
                !response.ok ||
                !result.success
            ) {

                throw new Error(
                    result.message ||
                    "Could not stop motors."
                );

            }


            showToast(
                "All motors stopped."
            );

        }

        catch (
            error
        ) {

            showToast(
                error.message,
                "error"
            );

        }

    }
);


/* ============================================================
   HTML ESCAPING
============================================================ */

function escapeHtml(
    value
) {

    return String(
        value
    )

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#039;"
        );

}


function escapeAttribute(
    value
) {

    return escapeHtml(
        value
    );

}


/* ============================================================
   START APPLICATION
============================================================ */

updateDetectionStatus();

loadMedicationSchedule();

loadMedicationLogs();


setInterval(
    updateDetectionStatus,
    500
);


setInterval(
    loadMedicationSchedule,
    10000
);


setInterval(
    loadMedicationLogs,
    10000
);
