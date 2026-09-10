(() => {
    "use strict";

    const progressBar = document.getElementById("reading-progress-bar");
    let progressTicking = false;

    const updateProgress = () => {
        const scrollable = document.documentElement.scrollHeight - window.innerHeight;
        const progress = scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0;
        progressBar.style.width = `${progress * 100}%`;
        progressTicking = false;
    };

    window.addEventListener("scroll", () => {
        if (!progressTicking) {
            window.requestAnimationFrame(updateProgress);
            progressTicking = true;
        }
    }, { passive: true });
    updateProgress();

    const gallery = document.getElementById("qualitative-gallery");
    const galleryImage = document.getElementById("gallery-image");
    const galleryTitle = document.getElementById("gallery-title");
    const galleryCount = document.getElementById("gallery-count");
    const galleryOpen = document.getElementById("gallery-open");
    const galleryTabs = Array.from(document.querySelectorAll("[data-gallery-index]"));

    const galleryItems = [
        {
            src: "assets/qual-1.webp",
            title: "Qualitative panel 1",
            tab: "Cases 1–2",
            alt: "Qualitative panel 1 comparing input, baseline, boundary priors, residual predictions, DERA results, and multiscale edge features"
        },
        {
            src: "assets/qual-2.webp",
            title: "Qualitative panel 2",
            tab: "Cases 3–4",
            alt: "Qualitative panel 2 comparing input, baseline, boundary priors, residual predictions, DERA results, and multiscale edge features"
        },
        {
            src: "assets/qual-3.webp",
            title: "Qualitative panel 3",
            tab: "Cases 5–6",
            alt: "Qualitative panel 3 comparing input, baseline, boundary priors, residual predictions, DERA results, and multiscale edge features"
        },
        {
            src: "assets/qual-4.webp",
            title: "Qualitative panel 4",
            tab: "Cases 7–8",
            alt: "Qualitative panel 4 comparing input, baseline, boundary priors, residual predictions, DERA results, and multiscale edge features"
        }
    ];

    let galleryIndex = 0;
    let galleryRequest = 0;

    const setGallery = (nextIndex, focusTab = false) => {
        const normalized = (nextIndex + galleryItems.length) % galleryItems.length;
        const item = galleryItems[normalized];
        const request = ++galleryRequest;
        galleryIndex = normalized;

        galleryImage.classList.add("is-switching");
        const preload = new Image();
        preload.onload = () => {
            if (request !== galleryRequest) return;
            galleryImage.src = item.src;
            galleryImage.alt = item.alt;
            galleryTitle.textContent = item.title;
            galleryCount.textContent = `${normalized + 1} of ${galleryItems.length}`;
            galleryOpen.setAttribute("aria-label", `Open ${item.title} full size`);

            galleryTabs.forEach((tab, index) => {
                const selected = index === normalized;
                tab.setAttribute("aria-selected", String(selected));
                tab.tabIndex = selected ? 0 : -1;
            });

            galleryImage.classList.remove("is-switching");
            if (focusTab) galleryTabs[normalized].focus();
        };
        preload.onerror = () => {
            if (request === galleryRequest) galleryImage.classList.remove("is-switching");
        };
        preload.src = item.src;
    };

    document.getElementById("gallery-prev").addEventListener("click", () => setGallery(galleryIndex - 1));
    document.getElementById("gallery-next").addEventListener("click", () => setGallery(galleryIndex + 1));
    galleryTabs.forEach((tab) => {
        tab.setAttribute("aria-controls", "gallery-panel");
        tab.addEventListener("click", () => setGallery(Number(tab.dataset.galleryIndex)));
    });

    gallery.addEventListener("keydown", (event) => {
        if (event.key === "ArrowLeft") {
            event.preventDefault();
            setGallery(galleryIndex - 1, event.target.matches("[role='tab']"));
        } else if (event.key === "ArrowRight") {
            event.preventDefault();
            setGallery(galleryIndex + 1, event.target.matches("[role='tab']"));
        } else if (event.key === "Home") {
            event.preventDefault();
            setGallery(0, event.target.matches("[role='tab']"));
        } else if (event.key === "End") {
            event.preventDefault();
            setGallery(galleryItems.length - 1, event.target.matches("[role='tab']"));
        }
    });

    let pointerStartX = null;
    gallery.addEventListener("pointerdown", (event) => {
        if (event.pointerType !== "mouse") pointerStartX = event.clientX;
    }, { passive: true });
    gallery.addEventListener("pointerup", (event) => {
        if (pointerStartX === null) return;
        const distance = event.clientX - pointerStartX;
        pointerStartX = null;
        if (Math.abs(distance) > 55) setGallery(galleryIndex + (distance < 0 ? 1 : -1));
    }, { passive: true });
    gallery.addEventListener("pointercancel", () => { pointerStartX = null; });

    const lightbox = document.getElementById("figure-lightbox");
    const lightboxImage = document.getElementById("lightbox-image");
    const lightboxClose = document.getElementById("lightbox-close");
    let lightboxOpener = null;

    const openLightbox = (src, alt, opener) => {
        lightboxOpener = opener;
        lightboxImage.src = src;
        lightboxImage.alt = alt;
        lightbox.showModal();
        lightboxClose.focus();
    };

    document.querySelectorAll("[data-lightbox-src]").forEach((button) => {
        button.addEventListener("click", () => {
            openLightbox(button.dataset.lightboxSrc, button.dataset.lightboxAlt || "Expanded research figure", button);
        });
    });

    galleryOpen.addEventListener("click", () => {
        const item = galleryItems[galleryIndex];
        openLightbox(item.src, item.alt, galleryOpen);
    });

    lightboxClose.addEventListener("click", () => lightbox.close());
    lightbox.addEventListener("click", (event) => {
        if (event.target === lightbox) lightbox.close();
    });
    lightbox.addEventListener("close", () => {
        lightboxImage.removeAttribute("src");
        lightboxImage.alt = "";
        if (lightboxOpener?.isConnected) lightboxOpener.focus();
        lightboxOpener = null;
    });

    document.querySelectorAll(".benchmark-details").forEach((details) => {
        const label = details.querySelector("summary span");
        details.addEventListener("toggle", () => {
            label.textContent = details.open ? "Collapse table" : "Expand table";
        });
    });
})();
