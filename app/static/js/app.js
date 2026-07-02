(function () {
    const toastIcons = {
        success: 'check-circle-2',
        error: 'circle-alert',
        warning: 'triangle-alert',
        info: 'info'
    };

    function ensureToastRegion() {
        let region = document.getElementById('toastRegion');
        if (!region) {
            region = document.createElement('div');
            region.id = 'toastRegion';
            region.className = 'toast-region';
            region.setAttribute('aria-live', 'polite');
            region.setAttribute('aria-atomic', 'true');
            document.body.appendChild(region);
        }
        return region;
    }

    function showToast(message, options = {}) {
        const variant = options.variant || 'info';
        const title = options.title || {
            success: 'Basarili',
            error: 'Hata',
            warning: 'Dikkat',
            info: 'Bilgi'
        }[variant] || 'Bilgi';
        const timeout = options.timeout ?? 4200;
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.dataset.variant = variant;
        toast.innerHTML = `
            <i data-lucide="${toastIcons[variant] || toastIcons.info}" class="toast-icon"></i>
            <div>
                <div class="toast-title">${escapeHtml(title)}</div>
                <div class="toast-message">${escapeHtml(message || '')}</div>
            </div>
            <button type="button" class="toast-close" aria-label="Bildirimi kapat">
                <i data-lucide="x" class="w-4 h-4"></i>
            </button>
        `;
        toast.querySelector('.toast-close').addEventListener('click', () => toast.remove());
        ensureToastRegion().appendChild(toast);
        if (window.lucide) window.lucide.createIcons();
        if (timeout) window.setTimeout(() => toast.remove(), timeout);
        return toast;
    }

    async function api(path, options = {}) {
        try {
            const res = await fetch(path, options);
            if (res.status === 401) {
                location.href = '/login';
                return res;
            }
            if (!res.ok && options.toast !== false) {
                const message = await responseMessage(res);
                showToast(message || `${path} HTTP ${res.status}`, {variant: 'error', title: 'API hatasi'});
            }
            return res;
        } catch (error) {
            if (options.toast !== false) {
                showToast(error.message || 'Baglanti hatasi olustu.', {variant: 'error', title: 'Baglanti hatasi'});
            }
            throw error;
        }
    }

    async function responseMessage(res) {
        const clone = res.clone();
        try {
            const data = await clone.json();
            return data.detail || data.error || data.message || JSON.stringify(data);
        } catch (_) {
            try {
                return await res.text();
            } catch (_) {
                return '';
            }
        }
    }

    function confirmAction(message, options = {}) {
        const ok = window.confirm(message);
        if (!ok && options.cancelToast) {
            showToast(options.cancelToast, {variant: 'info'});
        }
        return ok;
    }

    function emptyState(message, icon = 'inbox', actionHtml = '') {
        return `
            <div class="state-box">
                <div>
                    <i data-lucide="${icon}"></i>
                    <div class="font-semibold text-slate-700">${escapeHtml(message)}</div>
                    ${actionHtml ? `<div class="mt-4">${actionHtml}</div>` : ''}
                </div>
            </div>
        `;
    }

    function errorState(message, retry = '') {
        return `
            <div class="state-box text-rose-700">
                <div>
                    <i data-lucide="triangle-alert"></i>
                    <div class="font-semibold">${escapeHtml(message)}</div>
                    ${retry ? `<button onclick="${retry}" class="btn-secondary btn-icon px-4 py-2 mt-4"><i data-lucide="refresh-cw"></i>Tekrar dene</button>` : ''}
                </div>
            </div>
        `;
    }

    function skeletonCards(count = 3) {
        return Array.from({length: count}).map(() => '<div class="skeleton-card"></div>').join('');
    }

    function initMobileSidebar() {
        const toggle = document.getElementById('sidebarToggle');
        const close = document.getElementById('sidebarOverlay');
        if (!toggle || !close) return;
        toggle.addEventListener('click', () => {
            document.body.classList.toggle('sidebar-open');
            toggle.setAttribute('aria-expanded', document.body.classList.contains('sidebar-open') ? 'true' : 'false');
        });
        close.addEventListener('click', () => {
            document.body.classList.remove('sidebar-open');
            toggle.setAttribute('aria-expanded', 'false');
        });
        document.querySelectorAll('.app-sidebar a').forEach((link) => {
            link.addEventListener('click', () => {
                document.body.classList.remove('sidebar-open');
                toggle.setAttribute('aria-expanded', 'false');
            });
        });
    }

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, char => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }[char]));
    }

    window.api = api;
    window.showToast = showToast;
    window.confirmAction = confirmAction;
    window.emptyState = emptyState;
    window.errorState = errorState;
    window.skeletonCards = skeletonCards;
    window.escapeHtmlShared = escapeHtml;

    document.addEventListener('DOMContentLoaded', initMobileSidebar);
})();
