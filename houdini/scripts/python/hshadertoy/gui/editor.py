"""
hShadertoy Editor - Shadertoy IDE for Houdini
Imports and edits Shadertoy shaders for conversion to Houdini Copernicus
"""

import json
import os

import hou
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
                               QPushButton, QLabel, QTabWidget, QPlainTextEdit,
                               QComboBox, QMessageBox, QDialog, QMenu, QFileDialog,
                               QToolButton, QCheckBox)
from PySide6.QtGui import QFont

from hshadertoy.api.shadertoy import App
from hshadertoy.builder import build_shadertoy_hda

# Media library path
def get_media_json_path():
    """Get media.json path from $HSHADERTOY_HOUDINI or relative location"""
    hshadertoy_path = os.environ.get('HSHADERTOY_HOUDINI')
    if hshadertoy_path:
        path = os.path.join(hshadertoy_path, 'pic', 'media.json')
        if os.path.exists(path):
            return path

class ChannelSelector(QComboBox):
    """Dropdown for selecting texture channels (iChannel0-3)"""

    def __init__(self, channel_index, parent=None):
        super().__init__(parent)
        self.channel_index = channel_index
        self.media_library = self.load_media_library()
        self.populate_menu()

    def load_media_library(self):
        """Load media library from JSON"""
        MEDIA_JSON_PATH = os.environ.get('MEDIA_JSON_PATH') or get_media_json_path()
        if not MEDIA_JSON_PATH or not os.path.exists(MEDIA_JSON_PATH):
            print(f"Warning: Could not find media.json at {MEDIA_JSON_PATH}")
            return []

        try:
            with open(MEDIA_JSON_PATH, 'r') as f:
                return json.load(f).get('inputs', [])
        except Exception as e:
            print(f"Warning: Could not load media library: {e}")
            return []

    def populate_menu(self):
        """Populate dropdown with media options organized by folder structure from ichannel.json"""
        self.addItem("None", None)

        # Organize by folder structure from hda.folder.label
        folders = {}
        for item in self.media_library:
            folder_label = item.get('hda', {}).get('folder', {}).get('label', 'Unknown')
            if folder_label not in folders:
                folders[folder_label] = []
            folders[folder_label].append(item)

        # Add items by folder
        for folder_name in sorted(folders.keys()):
            items = folders[folder_name]
            if items:
                self.addItem(f"--- {folder_name} ---", None)
                # Sort by asset label
                items_sorted = sorted(items, key=lambda x: x.get('hda', {}).get('asset', {}).get('label', ''))
                for item in items_sorted:
                    asset_label = item.get('hda', {}).get('asset', {}).get('label', 'Unknown')
                    self.addItem(f"  {asset_label}", item)

    def get_selected_input(self):
        """Get the selected input data"""
        return self.currentData()

    def set_from_api_input(self, api_input):
        """Set selection based on API input data"""
        if not api_input:
            self.setCurrentIndex(0)
            return

        input_id = api_input.get('id')
        for i in range(self.count()):
            data = self.itemData(i)
            if data and data.get('id') == input_id:
                self.setCurrentIndex(i)
                return

        # If not found, set to None
        self.setCurrentIndex(0)


class SamplerOptionsPopup(QDialog):
    """Popup dialog for per-channel sampler settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sampler Options")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Popup)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Filter combo
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(4)
        filter_layout.addWidget(QLabel("Filter"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["mipmap", "linear", "nearest"])
        filter_layout.addWidget(self.filter_combo, 1)
        layout.addLayout(filter_layout)

        # Wrap combo
        wrap_layout = QHBoxLayout()
        wrap_layout.setContentsMargins(0, 0, 0, 0)
        wrap_layout.setSpacing(4)
        wrap_layout.addWidget(QLabel("Wrap"))
        self.wrap_combo = QComboBox()
        self.wrap_combo.addItems(["repeat", "clamp"])
        wrap_layout.addWidget(self.wrap_combo, 1)
        layout.addLayout(wrap_layout)

        # VFlip checkbox
        vflip_layout = QHBoxLayout()
        vflip_layout.setContentsMargins(0, 0, 0, 0)
        vflip_layout.setSpacing(4)
        self.vflip_checkbox = QCheckBox("VFlip")
        vflip_layout.addWidget(self.vflip_checkbox)
        vflip_layout.addStretch()
        layout.addLayout(vflip_layout)

        # OK button
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

    def set_values(self, sampler_state):
        """Populate controls from sampler state dict."""
        self.filter_combo.setCurrentText(sampler_state.get('filter', 'mipmap'))
        self.wrap_combo.setCurrentText(sampler_state.get('wrap', 'repeat'))
        vflip_value = str(sampler_state.get('vflip', 'false')).lower()
        self.vflip_checkbox.setChecked(vflip_value in ('true', '1', 'yes'))

    def get_values(self):
        """Return updated sampler values."""
        return {
            'filter': self.filter_combo.currentText(),
            'wrap': self.wrap_combo.currentText(),
            'vflip': 'true' if self.vflip_checkbox.isChecked() else 'false'
        }


class RenderPassTab(QWidget):
    """Widget for a single render pass (Image, Buffer A, etc.)"""

    def __init__(self, pass_name, pass_data=None, parent=None):
        super().__init__(parent)
        self.pass_name = pass_name
        self.pass_data = pass_data or {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Shader code editor
        self.code_editor = QPlainTextEdit()
        self.code_editor.setFont(QFont("Consolas", 10))
        self.code_editor.setPlaceholderText("void mainImage( out vec4 fragColor, in vec2 fragCoord )\n{\n    // Normalized pixel coordinates (from 0 to 1)\n    vec2 uv = fragCoord/iResolution.xy;\n\n    // Time varying pixel color\n    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));\n\n    // Output to screen\n    fragColor = vec4(col,1.0);\n}")

        # Set code from pass data
        code = self.pass_data.get('code', '')
        self.code_editor.setPlainText(code)

        layout.addWidget(self.code_editor, 1)

        # iChannel inputs
        channels_layout = QHBoxLayout()
        self.channel_selectors = []
        self.sampler_states = []

        inputs_by_channel = {
            input_data.get('channel'): input_data
            for input_data in self.pass_data.get('inputs', [])
            if isinstance(input_data, dict) and 'channel' in input_data
        }

        for i in range(4):
            channel_widget = QWidget()
            channel_layout = QVBoxLayout(channel_widget)
            channel_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"iChannel{i}")
            selector = ChannelSelector(i)

            sampler_state = self._default_sampler_state()

            input_data = inputs_by_channel.get(i)
            if input_data:
                selector.set_from_api_input(input_data)
                sampler_state = self._update_sampler_state_from_api(sampler_state, input_data.get('sampler', {}))

            selector_row = QHBoxLayout()
            selector_row.setContentsMargins(0, 0, 0, 0)
            selector_row.setSpacing(4)
            selector_row.addWidget(selector, 1)

            sampler_btn = QToolButton()
            sampler_btn.setText("...")
            sampler_btn.setToolTip("Sampler options")
            sampler_btn.setAutoRaise(True)
            sampler_btn.clicked.connect(lambda checked=False, idx=i, btn=sampler_btn: self.open_sampler_popup(idx, btn))
            selector_row.addWidget(sampler_btn)
            selector_row.addStretch()

            channel_layout.addWidget(label)
            channel_layout.addLayout(selector_row)
            channel_layout.addStretch()
            channels_layout.addWidget(channel_widget)

            self.channel_selectors.append(selector)
            self.sampler_states.append(sampler_state)

        layout.addLayout(channels_layout)

    def get_code(self):
        """Get shader code"""
        return self.code_editor.toPlainText()

    def get_inputs(self):
        """Get selected inputs as API format"""
        inputs = []
        for i, selector in enumerate(self.channel_selectors):
            selected = selector.get_selected_input()
            if selected:
                sampler_state = self.sampler_states[i]
                sampler = {
                    'filter': sampler_state.get('filter', 'mipmap'),
                    'wrap': sampler_state.get('wrap', 'repeat'),
                    'vflip': sampler_state.get('vflip', 'false'),
                    'srgb': sampler_state.get('srgb', 'false'),
                    'internal': sampler_state.get('internal', 'byte')
                }

                inputs.append({
                    'id': selected.get('id'),
                    'src': selected.get('src'),
                    'ctype': selected.get('ctype'),
                    'channel': i,
                    'sampler': sampler
                })
        return inputs

    def get_renderpass_data(self):
        """Get complete render pass data in API format"""
        return {
            'name': self.pass_name,
            'type': self.pass_name.lower().replace(' ', ''),
            'code': self.get_code(),
            'inputs': self.get_inputs(),
            'outputs': self.pass_data.get('outputs', [])
        }

    def _default_sampler_state(self):
        """Return default sampler configuration for a channel."""
        return {
            'filter': 'mipmap',
            'wrap': 'repeat',
            'vflip': 'false',
            'srgb': 'false',
            'internal': 'byte'
        }

    def _update_sampler_state_from_api(self, state, sampler):
        """Merge sampler data from API into local state."""
        if not sampler:
            return state

        updated_state = state.copy()
        for key in ('filter', 'wrap', 'vflip', 'srgb', 'internal'):
            if key in sampler:
                value = sampler[key]
                if key == 'vflip':
                    value = 'true' if str(value).lower() in ('true', '1', 'yes') else 'false'
                else:
                    value = str(value)
                updated_state[key] = value
        return updated_state

    def open_sampler_popup(self, index, button):
        """Show sampler popup anchored to the provided button."""
        popup = SamplerOptionsPopup(self)
        popup.set_values(self.sampler_states[index])

        button_pos = button.mapToGlobal(QtCore.QPoint(0, button.height()))
        popup.move(button_pos)

        if popup.exec_() == QDialog.Accepted:
            updated = popup.get_values()
            self.sampler_states[index].update(updated)


class ShadertoyEditor(QDialog):
    """Main Shadertoy Editor window"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shader_data = None
        self.api_key = os.environ['SHADERTOY_API_KEY']
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("hShadertoy Editor")
        self.resize(1000, 700)

        layout = QVBoxLayout(self)

        # Import section
        import_layout = QHBoxLayout()

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Shadertoy URL or ID (e.g., https://www.shadertoy.com/view/4l2XWw or 4l2XWw)")
        import_layout.addWidget(self.url_input, 1)

        import_btn = QPushButton("Import")
        import_btn.clicked.connect(self.import_shader)
        import_layout.addWidget(import_btn)

        layout.addLayout(import_layout)

        # Shader name
        self.name_label = QLabel("Shader Name: (none)")
        self.name_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.name_label)

        # Tab widget for render passes
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        layout.addWidget(self.tab_widget, 1)

        # Add default Image tab
        self.add_renderpass_tab("Image")

        # Bottom controls
        bottom_layout = QHBoxLayout()

        # Add tab menu
        add_tab_btn = QPushButton("+ Add Tab")
        add_tab_menu = QMenu(self)
        for tab_name in ["Common", "Buffer A", "Buffer B", "Buffer C", "Buffer D", "Cube A", "Sound"]:
            action = add_tab_menu.addAction(tab_name)
            # Use functools.partial or a factory function to avoid closure issue
            action.triggered.connect(lambda checked=False, name=tab_name: self.add_renderpass_tab(name))
        add_tab_btn.setMenu(add_tab_menu)
        bottom_layout.addWidget(add_tab_btn)

        bottom_layout.addStretch()

        # Mode selector
        bottom_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Transpiler", "Template", "Save to File"])
        bottom_layout.addWidget(self.mode_combo)

        # Build button
        build_btn = QPushButton("Build")
        build_btn.clicked.connect(self.build_shader)
        build_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 5px 15px; }")
        bottom_layout.addWidget(build_btn)

        layout.addLayout(bottom_layout)

    def extract_shader_id(self, url_or_id):
        """Extract shader ID from URL or return the ID itself"""
        url_or_id = url_or_id.strip()

        # If it's a full URL
        if 'shadertoy.com' in url_or_id:
            # Extract ID from URL like https://www.shadertoy.com/view/4l2XWw
            parts = url_or_id.split('/view/')
            if len(parts) > 1:
                return parts[1].split('?')[0].split('#')[0]

        # Otherwise assume it's already an ID
        return url_or_id

    def import_shader(self):
        """Import shader from Shadertoy API or JSON file"""
        url_or_id = self.url_input.text().strip()
        if not url_or_id:
            QMessageBox.warning(self, "Input Required", "Please enter a Shadertoy URL, ID, or JSON file path")
            return

        try:
            # Check if it's a file path
            if url_or_id.endswith('.json') and os.path.isfile(url_or_id):
                # Load from JSON file
                with open(url_or_id, 'r') as f:
                    data = json.load(f)

                # Check if it's a Shadertoy API format
                if 'Shader' in data:
                    shader_data = data['Shader']
                else:
                    shader_data = data

                # Store shader data
                self.shader_data = shader_data

                # Update UI
                self.populate_from_shader_data(shader_data)

                shader_name = shader_data.get('info', {}).get('name', 'Unknown')
                QMessageBox.information(self, "Success", f"Imported shader from file: {shader_name}")

            else:
                # Import from Shadertoy API
                shader_id = self.extract_shader_id(url_or_id)

                # Initialize API
                app = App(key=self.api_key)

                # Fetch shader
                QMessageBox.information(self, "Importing", f"Fetching shader {shader_id}...")
                shader_data = app.get_shader(shader_id)

                # Store shader data
                self.shader_data = shader_data

                # Update UI
                self.populate_from_shader_data(shader_data)

                QMessageBox.information(self, "Success", f"Imported shader: {shader_data['info']['name']}")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import shader:\n{str(e)}")

    def populate_from_shader_data(self, shader_data):
        """Populate editor from shader data"""
        # Update name
        shader_name = shader_data.get('info', {}).get('name', 'Unknown')
        self.name_label.setText(f"Shader Name: {shader_name}")

        # Clear existing tabs
        self.tab_widget.clear()

        # Add render passes
        for renderpass in shader_data.get('renderpass', []):
            pass_name = renderpass.get('name', 'Unknown')
            self.add_renderpass_tab(pass_name, renderpass)

    def add_renderpass_tab(self, name, pass_data=None):
        """Add a new render pass tab"""
        # Check if tab already exists
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == name:
                QMessageBox.warning(self, "Tab Exists", f"Tab '{name}' already exists")
                return

        tab = RenderPassTab(name, pass_data)
        index = self.tab_widget.addTab(tab, name)

        # Image tab cannot be closed
        if name == "Image":
            self.tab_widget.tabBar().setTabButton(index, QtWidgets.QTabBar.RightSide, None)

    def close_tab(self, index):
        """Close a tab (except Image)"""
        if self.tab_widget.tabText(index) == "Image":
            QMessageBox.warning(self, "Cannot Close", "The Image tab cannot be closed")
            return

        self.tab_widget.removeTab(index)

    def get_shader_json(self):
        """Get current shader data as JSON"""
        renderpasses = []

        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if isinstance(tab, RenderPassTab):
                renderpasses.append(tab.get_renderpass_data())

        shader_json = {
            'Shader': {
                'ver': '0.1',
                'info': self.shader_data.get('info', {}) if self.shader_data else {
                    'name': 'Custom Shader',
                    'username': 'hShadertoy',
                    'description': 'Created with hShadertoy Editor'
                },
                'renderpass': renderpasses
            }
        }

        return shader_json

    def build_shader(self):
        """Build shader using the selected mode."""
        mode = self.mode_combo.currentText()

        # Get shader JSON
        shader_json = self.get_shader_json()
        json_str = json.dumps(shader_json, indent=2)

        # Handle "Save to File" mode
        if mode == "Save to File":
            # Use QFileDialog instead of hou.ui.selectFile to avoid freezing
            start_dir = hou.expandString("$HOUDINI_USER_PREF_DIR") if hou else ""
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Shader JSON",
                start_dir,
                "JSON Files (*.json)"
            )

            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        f.write(json_str)
                    QMessageBox.information(self, "Success", f"Shader saved to:\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
            return

        # Build using the new builder
        # Handle both "Template" and "Transpiler" modes
        try:
            # Build the HDA
            # Don't pass parent_node - let builder create its own copnet
            # This follows Houdini convention where examples load into dedicated networks
            node = build_shadertoy_hda(
                shader_json=shader_json,
                mode=mode
            )

            QMessageBox.information(
                self,
                "Build Complete",
                f"Shader built with mode '{mode}' and created at:\n{node.path()}",
            )

        except Exception as exc:
            import traceback
            error_detail = traceback.format_exc()
            print(error_detail)
            QMessageBox.critical(
                self,
                "Build Error",
                f"Builder encountered an error:\n{exc}\n\nDetails:\n{error_detail}"
            )

    def _resolve_build_parent(self):
        """Find the target Houdini network for the builder."""
        if not hou:
            return None

        pane = hou.ui.paneTabUnderCursor()
        if not pane or not isinstance(pane, hou.NetworkEditor):
            pane = hou.ui.paneTabOfType(hou.paneTabType.NetworkEditor)

        if not pane:
            QMessageBox.warning(self, "No Network Editor", "Open a network editor to place the HDA.")
            return None

        return pane.pwd()


def show_editor(parent=None):
    """Show the Shadertoy Editor"""
    editor = ShadertoyEditor(parent)
    editor.exec_()
    return editor


# For Houdini integration
def run():
    """Run editor from Houdini"""
    parent = hou.qt.mainWindow()
    show_editor(parent)


